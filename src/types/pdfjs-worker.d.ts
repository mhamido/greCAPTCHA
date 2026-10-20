// pdfjs-dist ships no typings for its worker entry point. Only its existence matters here: the
// module is handed to pdf.js through `globalThis.pdfjsWorker` (see `loadPdfJs`).
declare module "pdfjs-dist/legacy/build/pdf.worker.mjs" {
  export const WorkerMessageHandler: unknown;
}
