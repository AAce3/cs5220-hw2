#include "common.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <omp.h>
#include <vector>

// Apply the force from neighbor to particle
static inline void apply_force(particle_t &particle, const particle_t &neighbor) {
  double dx = neighbor.x - particle.x;
  double dy = neighbor.y - particle.y;
  double r2 = dx * dx + dy * dy;

  if (r2 > cutoff * cutoff) {
    return;
  }

  r2 = fmax(r2, min_r * min_r);
  double r = std::sqrt(r2);

  double coef = (1 - cutoff / r) / r2 / mass;
  particle.ax += coef * dx;
  particle.ay += coef * dy;
}

// Integrate the ODE
static inline void move(particle_t &p, double size) {
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

    num_cells_x_ = std::ceil(size / cell_size_);
    num_cells_y_ = std::ceil(size / cell_size_);

    total_cells_ = num_cells_x_ * num_cells_y_;

    count_.assign(total_cells_, 0);
    offset_.assign(total_cells_ + 1, 0);
    back_ptrs_.assign(total_cells_, 0);
    data_.clear();
  }

  int cells_x() const { return num_cells_x_; }
  int cells_y() const { return num_cells_y_; }
  int total() const { return total_cells_; }

  int cell_index(int cx, int cy) const { return cy * num_cells_x_ + cx; }
  int begin(int cell) const { return offset_[cell]; }
  int end(int cell) const { return offset_[cell + 1]; }
  int particle_at(int k) const { return data_[k]; }

  int get_cell_id(double x, double y) const {
    int cx = x / cell_size_;
    int cy = y / cell_size_;

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

  friend struct ThreadScratch;
  friend void build_locals(Bins &locals_ref, particle_t *particles, int num_particles);
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

struct ThreadScratch {
  std::vector<int> local_count;
  std::vector<int> local_write;

  std::vector<int> ghost_count;
  std::vector<int> ghost_write;

  void reset_sizes(int local_cells, int ghost_cells) {
    if (local_count.size() != local_cells) {
      local_count.assign(local_cells, 0);
      local_write.assign(local_cells, 0);
    }
    if (ghost_count.size() != ghost_cells) {
      ghost_count.assign(ghost_cells, 0);
      ghost_write.assign(ghost_cells, 0);
    }
  }
};

static std::vector<ThreadScratch> thread_scratch;

static inline void add_ghost_targets(int cx, int cy, uint8_t edge_flags,
                                     int *out_cells, int &out_len) {
  int nx = locals.cells_x();
  int ny = locals.cells_y();
  int base = cy * nx + cx;

  out_len = 0;

  // left
  if ((edge_flags & 0x1) && cx > 0) {
    out_cells[out_len++] = base - 1;
  }

  // right
  if ((edge_flags & 0x2) && cx + 1 < nx) {
    out_cells[out_len++] = base + 1;
  }

  // bottom
  if ((edge_flags & 0x4) && cy > 0) {
    out_cells[out_len++] = base - nx;
  }

  // top
  if ((edge_flags & 0x8) && cy + 1 < ny) {
    out_cells[out_len++] = base + nx;
  }

  // bottom left
  if ((edge_flags & 0x5) == 0x5 && cx > 0 && cy > 0) {
    out_cells[out_len++] = base - 1 - nx;
  }

  // top left
  if ((edge_flags & 0x9) == 0x9 && cx > 0 && cy + 1 < ny) {
    out_cells[out_len++] = base - 1 + nx;
  }

  // bottom right
  if ((edge_flags & 0x6) == 0x6 && cx + 1 < nx && cy > 0) {
    out_cells[out_len++] = base + 1 - nx;
  }

  // top right
  if ((edge_flags & 0xA) == 0xA && cx + 1 < nx && cy + 1 < ny) {
    out_cells[out_len++] = base + 1 + nx;
  }
}

void build_locals(Bins &locals_ref, particle_t *particles, int num_particles) {
  int tid = omp_get_thread_num();
  int threads = omp_get_num_threads();

#pragma omp single
  {
    if (thread_scratch.size() != threads) {
      thread_scratch.resize(threads);
    }
    int local_cells = locals.total();
    int ghost_cells = ghosts.total();
    for (auto &ts : thread_scratch) {
      ts.reset_sizes(local_cells, ghost_cells);
    }

    if (ghost_scratch.ghost_cell.size() != num_particles) {
      ghost_scratch.ghost_cell.resize(num_particles);
    }
  }

  std::fill(thread_scratch[tid].local_count.begin(), thread_scratch[tid].local_count.end(), 0);

#pragma omp for schedule(static)
  for (int i = 0; i < num_particles; ++i) {
    int c = locals_ref.get_cell_id(particles[i].x, particles[i].y);
    ghost_scratch.ghost_cell[i] = c;
    thread_scratch[tid].local_count[c] += 1;
  }

#pragma omp for schedule(static)
  for (int c = 0; c < locals_ref.total_cells_; ++c) {
    int sum = 0;
    for (int t = 0; t < threads; ++t) {
      sum += thread_scratch[t].local_count[c];
    }
    locals_ref.count_[c] = sum;
  }

#pragma omp single
  {
    locals_ref.offset_[0] = 0;
    for (int c = 0; c < locals_ref.total_cells_; ++c) {
      locals_ref.offset_[c + 1] = locals_ref.offset_[c] + locals_ref.count_[c];
    }
    locals_ref.data_.resize(locals_ref.offset_[locals_ref.total_cells_]);
  }

#pragma omp for schedule(static)
  for (int c = 0; c < locals_ref.total_cells_; ++c) {
    int base = locals_ref.offset_[c];
    for (int t = 0; t < threads; ++t) {
      thread_scratch[t].local_write[c] = base;
      base += thread_scratch[t].local_count[c];
    }
  }

#pragma omp for schedule(static)
  for (int i = 0; i < num_particles; ++i) {
    int c = ghost_scratch.ghost_cell[i];
    int pos = thread_scratch[tid].local_write[c]++;
    locals_ref.data_[pos] = i;
  }
}

void build_ghosts(Bins &ghosts_ref, const Bins &locals_ref,
                  particle_t *particles, int num_particles) {
  int tid = omp_get_thread_num();
  int threads = omp_get_num_threads();

#pragma omp single
  {
    if (ghost_scratch.ghost_flags.size() != num_particles) {
      ghost_scratch.ghost_flags.resize(num_particles);
    }
  }

  std::fill(thread_scratch[tid].ghost_count.begin(), thread_scratch[tid].ghost_count.end(), 0);

#pragma omp for schedule(static)
  for (int i = 0; i < num_particles; ++i) {
    int home = ghost_scratch.ghost_cell[i];

    int cx = home % locals_ref.num_cells_x_;
    int cy = home / locals_ref.num_cells_x_;

    double x0 = cx * cell_size;
    double x1 = x0 + cell_size;
    double y0 = cy * cell_size;
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

    if (!edge_flags) {
      continue;
    }

    int targets[8];
    int len = 0;
    add_ghost_targets(cx, cy, edge_flags, targets, len);

    for (int k = 0; k < len; ++k) {
      thread_scratch[tid].ghost_count[targets[k]] += 1;
    }
  }

#pragma omp for schedule(static)
  for (int c = 0; c < ghosts_ref.total_cells_; ++c) {
    int sum = 0;
    for (int t = 0; t < threads; ++t) {
      sum += thread_scratch[t].ghost_count[c];
    }
    ghosts_ref.count_[c] = sum;
  }

#pragma omp single
  {
    ghosts_ref.offset_[0] = 0;
    for (int c = 0; c < ghosts_ref.total_cells_; ++c) {
      ghosts_ref.offset_[c + 1] = ghosts_ref.offset_[c] + ghosts_ref.count_[c];
    }
    ghosts_ref.data_.resize(ghosts_ref.offset_[ghosts_ref.total_cells_]);
  }

#pragma omp for schedule(static)
  for (int c = 0; c < ghosts_ref.total_cells_; ++c) {
    int base = ghosts_ref.offset_[c];
    for (int t = 0; t < threads; ++t) {
      thread_scratch[t].ghost_write[c] = base;
      base += thread_scratch[t].ghost_count[c];
    }
  }

#pragma omp for schedule(static)
  for (int i = 0; i < num_particles; ++i) {
    uint8_t mask = ghost_scratch.ghost_flags[i];
    if (mask == 0) {
      continue;
    }

    int home = ghost_scratch.ghost_cell[i];
    int cx = home % locals_ref.num_cells_x_;
    int cy = home / locals_ref.num_cells_x_;

    int targets[8];
    int len = 0;
    add_ghost_targets(cx, cy, mask, targets, len);

    for (int k = 0; k < len; ++k) {
      int cell = targets[k];
      int pos = thread_scratch[tid].ghost_write[cell]++;
      ghosts_ref.data_[pos] = i;
    }
  }
}

void init_simulation(particle_t *particles, int num_particles, double size) {
  (void)particles;

  ghost_width = cutoff;
  cell_size = 16 * cutoff;

  locals.init(size, cell_size);
  ghosts.init(size, cell_size);

  ghost_scratch.ghost_flags.clear();
  ghost_scratch.ghost_cell.clear();

  thread_scratch.clear();
}

void simulate_one_step(particle_t *particles, int num_particles, double size) {
#pragma omp for schedule(static)
  for (int i = 0; i < num_particles; ++i) {
    particles[i].ax = 0.0;
    particles[i].ay = 0.0;
  }

  build_locals(locals, particles, num_particles);
  build_ghosts(ghosts, locals, particles, num_particles);

  // forces
#pragma omp for collapse(2) schedule(static)
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
          apply_force(particles[j], particles[i]);
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
#pragma omp for schedule(static)
  for (int i = 0; i < num_particles; ++i) {
    move(particles[i], size);
  }
}

