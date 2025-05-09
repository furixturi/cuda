# Awesome Starting Point to Learn CUDA

## Resources

### Book
[Programming Massively Parallel Processors 4th edition (Amazon)](https://www.amazon.com/Programming-Massively-Parallel-Processors-Hands/dp/0323912311)


### Online course
[ECE 408 at University of Illinois Urbana-Champaign, summer 2024](https://lumetta.web.engr.illinois.edu/408-Sum24/)
Taught by Prof Steve Lumetta (lumetta@illinois.edu)

## Study plan per the book

This study plan is based on the structure used in courses like ECE408 and follows a **two-phase approach** aligned with the book *Programming Massively Parallel Processors*.

---

### 📘 Phase 1: Fundamentals and Basic Patterns

- **Book Sections**: Parts I & II
- **Duration**: ~7 weeks
- **Chapters Covered**: 2 to 12
- **Format**: Weekly lectures + weekly programming assignments

#### 🎯 Goals:
- Learn CUDA fundamentals and data-parallel programming
- Practice skills through guided assignments

#### 📅 Weekly Breakdown:

- **Week 1**
  - **Lecture**: Chapter 2 (CUDA memory/threading model, C language extensions, tools)
  - **Assignment**: Vector addition in CUDA

- **Weeks 2–3**
  - **Lectures**: Chapters 3–6 (CUDA memory model, thread execution model, GPU hardware performance, system architecture)
  - **Assignments**: Matrix-matrix multiplication (multiple implementations and optimizations)

- **Weeks 4–7**
  - **Lectures**: Chapters 7–12 (common data-parallel patterns)
    - Convolution
    - Histogram
    - Reduction
    - Prefix Sum (Scan)
  - **Assignments**: Weekly CUDA implementations of the above patterns

By the end of Phase 1, students are expected to be comfortable with basic parallel programming and able to work more independently.

---

### 🚀 Phase 2: Advanced Patterns and Final Project

- **Book Sections**: Parts III & IV
- **Duration**: ~5–6 weeks (varies depending on course structure)
- **Chapters**: May include Chapters 11, 14, 15 (Scan, Sparse Matrix, Graph Traversal)
- **Format**: Final project with weekly milestones (no weekly assignments)

#### 🎯 Goals:
- Apply skills in a real-world CUDA project
- Explore advanced programming patterns
- Learn practices for optimizing and finalizing applications

#### 🧩 Format and Flexibility:
- No fixed weekly assignments
- Final project includes weekly milestones
- Instructors may:
  - Skip or reorder chapters
  - Include guest lectures or paper discussions
  - Provide lecture support tailored to the project

---

### 📝 Summary Table

| Phase    | Focus                         | Duration   | Deliverables                  |
|----------|-------------------------------|------------|-------------------------------|
| Phase 1  | Fundamentals & Basic Patterns | ~7 weeks   | Weekly assignments            |
| Phase 2  | Advanced Topics & Final Project | ~5–6 weeks | Final project with milestones |