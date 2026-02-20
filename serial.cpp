#include "common.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

// Apply the force from neighbor to particle
void apply_force(particle_t &particle, particle_t &neighbor) {
  double dx = neighbor.x - particle.x;
  double dy = neighbor.y - particle.y;
  double r2 = dx * dx + dy * dy;

  if (r2 > cutoff * cutoff) {
    return;
  }

  r2 = fmax(r2, min_r * min_r);
  double r = sqrt(r2);

  double coef = (1 - cutoff / r) / r2 / mass;
  particle.ax += coef * dx;
  particle.ay += coef * dy;
}

// Integrate the ODE
void move(particle_t &p, double size) {
  p.vx += p.ax * dt;
  p.vy += p.ay * dt;
  p.x += p.vx * dt;
  p.y += p.vy * dt;

  while (p.x < 0 || p.x > size) {
    if (p.x < 0) {
      p.x = -p.x;
    } else {
      p.x = 2 * size - p.x;
    }
    p.vx = -p.vx;
  }

  while (p.y < 0 || p.y > size) {
    if (p.y < 0) {
      p.y = -p.y;
    } else {
      p.y = 2 * size - p.y;
    }
    p.vy = -p.vy;
  }
}

class Bins {
public:
  void init(double size, double cell_size) {
    cell_size_ = cell_size;

    num_cells_x_ = static_cast<int>(std::ceil(size / cell_size_));
    num_cells_y_ = static_cast<int>(std::ceil(size / cell_size_));

    total_cells_ = num_cells_x_ * num_cells_y_;

    count_.assign(total_cells_, 0);
    offset_.assign(total_cells_ + 1, 0);
    back_ptrs_.assign(total_cells_, 0);
    data_.clear();
  }

  void build_locals(particle_t *particles, int num_particles) {
    if (data_.size() != (size_t)num_particles) {
      data_.resize(num_particles);
    }

    std::fill(count_.begin(), count_.end(), 0);
    for (int i = 0; i < num_particles; ++i) {
      int c = cell_id_of(particles[i].x, particles[i].y);
      count_[c] += 1;
    }

    offset_[0] = 0;
    for (int c = 0; c < total_cells_; ++c) {
      offset_[c + 1] = offset_[c] + count_[c];
    }

    for (int c = 0; c < total_cells_; ++c) {
      back_ptrs_[c] = offset_[c];
    }

    for (int i = 0; i < num_particles; ++i) {
      int c = cell_id_of(particles[i].x, particles[i].y);
      int pos = back_ptrs_[c];
      back_ptrs_[c] += 1;
      data_[pos] = i;
    }
  }

  int cells_x() const { return num_cells_x_; }
  int cells_y() const { return num_cells_y_; }
  int total() const { return total_cells_; }

  int cell_index(int cx, int cy) const { return cy * num_cells_x_ + cx; }
  int begin(int cell) const { return offset_[cell]; }
  int end(int cell) const { return offset_[cell + 1]; }
  int particle_at(int k) const { return data_[k]; }

  int cell_id_of(double x, double y) const {
    int cx = static_cast<int>(x / cell_size_);
    int cy = static_cast<int>(y / cell_size_);

    if (cx < 0) {
      cx = 0;
    } else if (cx >= num_cells_x_) {
      cx = num_cells_x_ - 1;
    }

    if (cy < 0) {
      cy = 0;
    } else if (cy >= num_cells_y_) {
      cy = num_cells_y_ - 1;
    }

    return cy * num_cells_x_ + cx;
  }

private:
  int num_cells_x_ = 0;
  int num_cells_y_ = 0;
  int total_cells_ = 0;
  double cell_size_ = 0.0;

  std::vector<int> count_;
  std::vector<int> offset_;
  std::vector<int> back_ptrs_;
  std::vector<int> data_;

  friend void build_ghosts(Bins &ghosts_ref, const Bins &locals_ref,
                           particle_t *particles, int num_particles);
};

static double cell_size = 0.0;
static double ghost_width = 0.0;

static Bins locals;
static Bins ghosts;

struct GhostScratch {
  std::vector<uint8_t> ghost_flags;
  std::vector<int> ghost_cell;
};

static GhostScratch ghost_scratch;

static inline void add_ghost_targets(int cx, int cy, uint8_t edge_flags,
                                     int *out_cells, int &out_len) {
  int nx = locals.cells_x();
  int ny = locals.cells_y();
  int base = cy * nx + cx;

  out_len = 0;

  // left
  if ((edge_flags & 0x1) != 0) {
    if (cx > 0) {
      out_cells[out_len] = base - 1;
      out_len += 1;
    }
  }

  // right
  if ((edge_flags & 0x2) != 0) {
    if (cx + 1 < nx) {
      out_cells[out_len] = base + 1;
      out_len += 1;
    }
  }

  // bottom
  if ((edge_flags & 0x4) != 0) {
    if (cy > 0) {
      out_cells[out_len] = base - nx;
      out_len += 1;
    }
  }

  // top
  if ((edge_flags & 0x8) != 0) {
    if (cy + 1 < ny) {
      out_cells[out_len] = base + nx;
      out_len += 1;
    }
  }

  // bottom left
  if ((edge_flags & 0x5) == 0x5) {
    if (cx > 0 && cy > 0) {
      out_cells[out_len++] = base - 1 - nx;
    }
  }

  // top left
  if ((edge_flags & 0x9) == 0x9) {
    if (cx > 0 && cy + 1 < ny) {
      out_cells[out_len++] = base - 1 + nx;
    }
  }

  // bottom right
  if ((edge_flags & 0x6) == 0x6) {
    if (cx + 1 < nx && cy > 0) {
      out_cells[out_len++] = base + 1 - nx;
    }
  }

  // top right
  if ((edge_flags & 0xA) == 0xA) {
    if (cx + 1 < nx && cy + 1 < ny) {
      out_cells[out_len++] = base + 1 + nx;
    }
  }
}

void build_ghosts(Bins &ghosts_ref, const Bins &locals_ref,
                  particle_t *particles, int num_particles) {
  int total_cells = locals_ref.total();

  std::fill(ghosts_ref.count_.begin(), ghosts_ref.count_.end(), 0);

  if (ghost_scratch.ghost_flags.size() != (size_t)num_particles) {
    ghost_scratch.ghost_flags.resize(num_particles);
  }
  if (ghost_scratch.ghost_cell.size() != (size_t)num_particles) {
    ghost_scratch.ghost_cell.resize(num_particles);
  }

  int tmp_cells[8];
  int tmp_len = 0;

  for (int i = 0; i < num_particles; ++i) {
    int home = locals_ref.cell_id_of(particles[i].x, particles[i].y);
    ghost_scratch.ghost_cell[i] = home;

    int cx = home % locals_ref.cells_x();
    int cy = home / locals_ref.cells_x();

    double x0 = (double)cx * cell_size;
    double x1 = x0 + cell_size;
    double y0 = (double)cy * cell_size;
    double y1 = y0 + cell_size;

    uint8_t edge_flags = 0;

    if ((particles[i].x - x0) < ghost_width) {
      edge_flags |= 0x1;
    }
    if ((x1 - particles[i].x) < ghost_width) {
      edge_flags |= 0x2;
    }
    if ((particles[i].y - y0) < ghost_width) {
      edge_flags |= 0x4;
    }
    if ((y1 - particles[i].y) < ghost_width) {
      edge_flags |= 0x8;
    }

    ghost_scratch.ghost_flags[i] = edge_flags;

    add_ghost_targets(cx, cy, edge_flags, tmp_cells, tmp_len);

    for (int t = 0; t < tmp_len; ++t) {
      ghosts_ref.count_[tmp_cells[t]] += 1;
    }
  }

  ghosts_ref.offset_[0] = 0;
  for (int c = 0; c < total_cells; ++c) {
    ghosts_ref.offset_[c + 1] = ghosts_ref.offset_[c] + ghosts_ref.count_[c];
  }

  int total_ghosts = ghosts_ref.offset_[total_cells];
  if (ghosts_ref.data_.size() != (size_t)total_ghosts) {
    ghosts_ref.data_.resize(total_ghosts);
  }

  for (int c = 0; c < total_cells; ++c) {
    ghosts_ref.back_ptrs_[c] = ghosts_ref.offset_[c];
  }

  for (int i = 0; i < num_particles; ++i) {
    uint8_t mask = ghost_scratch.ghost_flags[i];
    if (mask == 0) {
      continue;
    }

    int home = ghost_scratch.ghost_cell[i];
    int cx = home % locals_ref.cells_x();
    int cy = home / locals_ref.cells_x();

    add_ghost_targets(cx, cy, mask, tmp_cells, tmp_len);

    for (int t = 0; t < tmp_len; ++t) {
      int cell = tmp_cells[t];
      int pos = ghosts_ref.back_ptrs_[cell];
      ghosts_ref.back_ptrs_[cell] += 1;
      ghosts_ref.data_[pos] = i;
    }
  }
}

void init_simulation(particle_t *particles, int num_particles, double size) {
  ghost_width = cutoff;
  cell_size = 10 * cutoff;

  locals.init(size, cell_size);
  ghosts.init(size, cell_size);
}

void simulate_one_step(particle_t *particles, int num_particles, double size) {
  // reset accelerations
  for (int i = 0; i < num_particles; ++i) {
    particles[i].ax = 0.0;
    particles[i].ay = 0.0;
  }

  locals.build_locals(particles, num_particles);
  build_ghosts(ghosts, locals, particles, num_particles);

  for (int cy = 0; cy < locals.cells_y(); ++cy) {
    for (int cx = 0; cx < locals.cells_x(); ++cx) {
      int cell = locals.cell_index(cx, cy);

      int begin = locals.begin(cell);
      int end = locals.end(cell);

      for (int a = begin; a < end; ++a) {
        int i = locals.particle_at(a);
        for (int b = a + 1; b < end; ++b) {
          int j = locals.particle_at(b);

          apply_force(particles[i], particles[j]);
          apply_force(particles[j], particles[i]); // equal and opposite
        }
      }

      for (int a = begin; a < end; ++a) {
        int i = locals.particle_at(a);
        for (int g = ghosts.begin(cell); g < ghosts.end(cell); ++g) {
          int j = ghosts.particle_at(g);
          apply_force(particles[i], particles[j]);
        }
      }
    }
  }

  // move
  for (int i = 0; i < num_particles; ++i) {
    move(particles[i], size);
  }
}
