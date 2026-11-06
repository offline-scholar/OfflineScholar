# Offline Scholar — Frontend

The user interface for Offline Scholar: a React + TypeScript single-page app, built with Vite and styled with
Tailwind CSS. It talks only to the local FastAPI backend under `/api`, so everything stays on your device.
The layout follows the Figma Make design.

For the backend, models and the desktop launcher, see the [main README](../README.md).

## Stack

| Piece        | Version | Used for                                 |
| ------------ | ------- | ---------------------------------------- |
| React        | 18.3    | UI                                       |
| TypeScript   | 5.6     | Types (`tsc -b` runs as part of `build`) |
| Vite         | 6.0     | Dev server and production build          |
| Tailwind CSS | 4.1     | Styling (through `@tailwindcss/vite`)    |
| lucide-react | 0.460   | Icons                                    |

The app has no router library and no state library. Routing uses the URL hash and state lives in a React context.

## Getting started

You need Node.js 18 or newer, and the backend running on port 8000 (see the main README).

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173
```

In development Vite forwards every `/api` request to `http://127.0.0.1:8000` (see `vite.config.ts`), so the UI
and backend behave as if they share one origin.

| Script            | What it does                                                   |
| ----------------- | -------------------------------------------------------------- |
| `npm run dev`     | Starts the dev server with hot reload on :5173                 |
| `npm run build`   | Type-checks, then builds the production bundle into `dist/`    |
| `npm run preview` | Serves the built `dist/` locally so you can check the build    |

### Production / desktop build

```bash
npm run build
```

FastAPI serves `frontend/dist` at `/` when that folder exists (`backend/main.py`). So after a build, running
`uvicorn backend.main:app --port 8000`, `python scripts/desktop.py` or `Offline Scholar.bat` opens the full app at
http://localhost:8000 with no separate frontend server.

## Project structure

```
frontend/
├── index.html            Vite entry HTML
├── vite.config.ts        React + Tailwind plugins, /api proxy to :8000
├── public/favicon.svg
└── src/
    ├── main.tsx          Mounts <App /> into #root
    ├── App.tsx           App shell: side rail, header, subject picker, offline banner, keyboard shortcuts
    ├── router.ts         Hash routes (#dashboard, #quiz, …) and hand-offs between pages
    ├── store.tsx         StoreProvider: subjects, documents, analysis, revision, health, saved settings
    ├── helpData.ts       Text for the Help & Shortcuts page
    ├── index.css         Tailwind import and global styles
    ├── components/ui.tsx Shared UI (Button, Card, cx helper, …)
    ├── lib/
    │   ├── api.ts        Typed client for every backend endpoint, plus ApiError
    │   └── format.ts     Formatting helpers
    └── pages/            One file per screen (see below)
```

## Pages

| Route           | File             | Purpose                                                                  |
| --------------- | ---------------- | ------------------------------------------------------------------------ |
| `#dashboard`    | `Dashboard.tsx`  | Overview of the current subject and its priority topics                  |
| `#documents`    | `Documents.tsx`  | Upload PDFs as lecture notes, past papers or syllabus; watch indexing; delete |
| `#assistant`    | `Assistant.tsx`  | Ask questions and get answers that cite your documents and pages         |
| `#papers`       | `PastPapers.tsx` | Topic frequency and questions that repeat across past papers             |
| `#coverage`     | `Coverage.tsx`   | How each syllabus topic is covered by your notes                         |
| `#quiz`         | `Quiz.tsx`       | Generate a practice quiz on a topic and submit your results              |
| `#revision`     | `Revision.tsx`   | Revision plan ranked by priority and mastery                             |
| `#settings`     | `Settings.tsx`   | Display name, default quiz length, whether to show sources               |
| `#help`         | `Help.tsx`       | Help articles and keyboard shortcuts                                     |

Pages can pass work to each other. For example, a topic on the Dashboard can open the Study Assistant with a
question already filled in (`askAssistant`), or start a quiz on that topic (`startFocusedQuiz`).

## Talking to the backend

All requests go through `src/lib/api.ts`, which wraps `fetch("/api" + path)` and turns errors into an `ApiError`.

| Method                                 | Endpoint                         |
| -------------------------------------- | -------------------------------- |
| `api.health()`                         | `GET /api/health`                |
| `api.subjects()` / `createSubject`     | `GET` / `POST /api/subjects`     |
| `api.deleteSubject(id)`                | `DELETE /api/subjects/{id}`      |
| `api.documents(subjectId)`             | `GET /api/documents?subject_id=` |
| `api.upload(file, subjectId, docType)` | `POST /api/documents/upload`     |
| `api.deleteDocument(id)`               | `DELETE /api/documents/{id}`     |
| `api.ask(question, subjectId)`         | `POST /api/chat/ask`             |
| `api.analysis(subjectId)`              | `GET /api/analysis/{subjectId}`  |
| `api.revision(subjectId)`              | `GET /api/revision/{subjectId}`  |
| `api.generateQuiz(subjectId, topic, n)`| `POST /api/quiz/generate`        |
| `api.submitQuiz(subjectId, answers)`   | `POST /api/quiz/submit`          |

The request and response types in `api.ts` match the backend schemas. See [docs/API.md](../docs/API.md) for the
full API reference. When the backend can't be reached, the store sets `backendDown` and the shell shows a banner.

## Local state

UI preferences are kept in the browser's `localStorage` under `offline-scholar-ui-v2`: the selected subject,
settings (name, quiz length, show sources), quiz history and completed revision sessions. Documents, indexes and
analysis live in the backend's `data/` folder, not in the browser. To reset the UI, clear that key.

## Keyboard shortcuts

| Keys            | Action                                                                          |
| --------------- | ------------------------------------------------------------------------------- |
| `Alt` + `1`–`7` | Go to Dashboard, Documents, Study Assistant, Past Papers, Coverage, Quiz, Revision |
| `Alt` + `H`     | Open Help & Shortcuts                                                           |
| `/`             | Focus the question box on the Study Assistant page                              |
| `Esc`           | Close the subject menu                                                          |

## Contributing

- Keep every network call in `src/lib/api.ts`, and update its types when a backend schema changes.
- Add a screen by creating `src/pages/<Name>.tsx`, adding its id to `Route` and `ROUTES` in `router.ts`, and
  registering it in `PAGES` in `App.tsx`.
- Run `npm run build` before opening a PR. It type-checks the whole app.
