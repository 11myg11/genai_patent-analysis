# GenAI - Patent Analysis

The project aims to build an AI-supported tool that understands patents, identifies potential design conflicts, and helps engineers create safer and more innovative product solutions.

---

## GitHub Guide

This project uses a **very simple Git workflow** so everyone can collaborate easily.

---

### Basic Idea

- We have **one main branch** → `main`
- Everyone works in their **own branch**
- When you're done → you **merge your work into** `main`

---

### Branch Structure

```
main
├── feature/patent-parser
├── feature/risk-analysis
└── feature/ui-dashboard
```

---

### The Workflow

#### 0. Clone the repo (first time only)

```
git clone https://github.com/your-org/your-repo.git
cd your-repo
```

#### 1. Get the latest version of `main`

Always start here:

```
git checkout main
git pull
```

#### 2. Create your own branch

```
git checkout -b feature/your-task-name
```

Example:

```
git checkout -b feature/risk-analysis
```

#### 3. Work and save your changes

```
git add .
git commit -m "what you did"
```

Example:

```
git commit -m "added risk scoring logic"
```

#### 4. Push your branch to GitHub

First pull to catch any conflicts, then push:

```
git pull origin main
git push origin feature/your-task-name
```

#### 5. Create a Pull Request (PR)

- Go to GitHub
- Click "Compare & pull request"
- Add a short description
- Ask someone to review

#### 6. Merge your work

After approval:

- Click Merge
- (Optional) Delete your branch

---

### Naming Rules

Keep names simple:

```
feature/<what-you-built>
fix/<bug-name>
docs/<topic>
```

Examples:

```
feature/data-processing
fix/login-error
docs/readme-update
```

---

### Important Rules

- Do not push directly to main
- Always create a branch
- Pull latest changes before starting
- Keep commits small and clear

---

### How We Work Together

- Everyone works in parallel
- Try not to work simultaneously on one feature — this will create conflicts
- We combine work through Pull Requests

---

### Quick Summary

```
1. git clone <repo-url>        ← first time only
2. git checkout main
3. git pull
4. git checkout -b feature/your-task
5. work + git add . + git commit -m "message"
6. git pull origin main
7. git push origin feature/your-task
8. create PR on GitHub
9. merge after review
```
