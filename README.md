# Denarii 🧠💰
**A 2025-ready tokenomics-first project environment**

This repo is built for Web3 builders, token designers, and startup accelerator teams who want:
- A clean, containerized JS dev environment
- Built-in linting, testing, and formatting
- AI-readiness with Codex and Codespaces support

---

## 🚀 Features

- ✅ Devcontainer setup (`devcontainer.json`)
- ✅ ESLint + Prettier + Jest configured
- ✅ Ready for GitHub Codespaces or ChatGPT Codex
- ✅ Auto-installs VS Code extensions

---

## 🛠️ Scripts

```bash
npm run lint     # Lint your JS code
npm run format   # Format code using Prettier
npm run test     # Run Jest tests
.
├── .eslintrc.json
├── .prettierrc
├── devcontainer.json
├── tests/
│   └── sample.test.js

---

Say the word and I’ll drop that into your project right now as a fully formatted `README.md`.

## 📈 Tokenomics Simulation

Run the Python simulation and generate a text report:

```bash
bash run.sh
```

The script outputs a simple simulation, performs risk analysis (using OpenAI if an API key is available),
and writes the combined results to `report.txt`.
