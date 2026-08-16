"""Interface gráfica simples para o classificador de faltas ATP."""

from __future__ import annotations

import json
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from infer_fault import infer_fault


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "classificador_config.json"
CLASS_NAMES = {
    "AG": "Fase A – terra",
    "BG": "Fase B – terra",
    "CG": "Fase C – terra",
    "AB": "Fases A – B",
    "BC": "Fases B – C",
    "CA": "Fases C – A",
    "ABG": "Fases A – B – terra",
    "BCG": "Fases B – C – terra",
    "CAG": "Fases C – A – terra",
    "ABC": "Falta trifásica",
}


LOW_CONFIDENCE_THRESHOLD = 0.30
NO_SIGNATURE_HINT = (
    "Confiança muito baixa: o sinal pode não conter uma falta real nesta janela "
    "(confira, no ATPDraw, se todas as chaves de falta fecham dentro de t_cl "
    "entre 0,0833 e 0,1000 s e se Rfault está ligado ao mesmo nó de terra)."
)


def _load_paths() -> tuple[Path, Path]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    classifier = Path(config["classifier"])
    freeze = Path(config["freeze"])
    if not freeze.is_absolute():
        freeze = BASE_DIR / freeze
    return classifier, freeze


def _freeze_summary(freeze_path: Path) -> str:
    try:
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return f"Congelamento: {freeze_path.name}"
    version = freeze.get("classifier", {}).get("sha256", "")[:8]
    classifier_range = freeze.get("classifier", {}).get("validated_distance_range_km")
    localizer_range = freeze.get("localizer", {}).get("validated_distance_range_km")
    parts = [f"Congelamento: {freeze_path.name} (classificador {version}…)"]
    if classifier_range:
        parts.append(f"classificação validada {classifier_range[0]}-{classifier_range[1]}km")
    if localizer_range:
        parts.append(f"localização validada {localizer_range[0]}-{localizer_range[1]}km")
    return " · ".join(parts)


class ClassifierApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Classificador de Faltas ATP")
        self.geometry("720x580")
        self.minsize(650, 520)
        self.selected_file: Path | None = None
        self.last_result: dict[str, object] | None = None
        self._build()

    def _build(self) -> None:
        frame = ttk.Frame(self, padding=22)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Classificador de Faltas ATP", font=("Segoe UI", 20, "bold")).pack(anchor="w")
        ttk.Label(
            frame,
            text="Escolha um arquivo PL4 com os 12 canais do circuito PDT–BEA.",
            font=("Segoe UI", 10),
        ).pack(anchor="w", pady=(4, 4))
        self.freeze_var = tk.StringVar(value="Carregando configuração…")
        ttk.Label(
            frame, textvariable=self.freeze_var, font=("Segoe UI", 8), foreground="#555555",
        ).pack(anchor="w", pady=(0, 14))
        self._refresh_freeze_summary()

        chooser = ttk.Frame(frame)
        chooser.pack(fill="x")
        self.file_var = tk.StringVar(value="Nenhum arquivo selecionado")
        ttk.Entry(chooser, textvariable=self.file_var, state="readonly").pack(side="left", fill="x", expand=True)
        ttk.Button(chooser, text="Escolher PL4…", command=self.choose_file).pack(side="left", padx=(10, 0))

        self.analyze_button = ttk.Button(frame, text="Analisar arquivo", command=self.start_analysis, state="disabled")
        self.analyze_button.pack(anchor="w", pady=16)
        self.progress = ttk.Progressbar(frame, mode="indeterminate")
        self.progress.pack(fill="x")

        result = ttk.LabelFrame(frame, text="Resultado", padding=16)
        result.pack(fill="both", expand=True, pady=(18, 0))
        self.class_var = tk.StringVar(value="Tipo de falta: —")
        self.vote_var = tk.StringVar(value="Confiança do conjunto: —")
        self.location_var = tk.StringVar(value="Distância desde PDT: —")
        self.snr_var = tk.StringVar(value="Qualidade estimada do sinal: —")
        self.warning_var = tk.StringVar(value="")
        ttk.Label(result, textvariable=self.class_var, font=("Segoe UI", 15, "bold")).pack(anchor="w", pady=3)
        ttk.Label(result, textvariable=self.vote_var).pack(anchor="w", pady=3)
        ttk.Label(result, textvariable=self.location_var, font=("Segoe UI", 12)).pack(anchor="w", pady=3)
        ttk.Label(result, textvariable=self.snr_var).pack(anchor="w", pady=3)
        ttk.Label(result, textvariable=self.warning_var, foreground="#9a4d00", wraplength=630).pack(anchor="w", pady=(10, 3))
        self.save_button = ttk.Button(result, text="Salvar resultado…", command=self.save_result, state="disabled")
        self.save_button.pack(anchor="w", pady=(12, 0))

    def _refresh_freeze_summary(self) -> None:
        try:
            _, freeze = _load_paths()
            self.freeze_var.set(_freeze_summary(freeze))
        except Exception:
            self.freeze_var.set("Não foi possível ler classificador_config.json")

    def choose_file(self) -> None:
        selected = filedialog.askopenfilename(title="Escolha o arquivo PL4", filetypes=(("Arquivo ATP PL4", "*.pl4"), ("Todos os arquivos", "*.*")))
        if selected:
            self.selected_file = Path(selected)
            self.file_var.set(selected)
            self.analyze_button.configure(state="normal")

    def start_analysis(self) -> None:
        if self.selected_file is None:
            return
        self.analyze_button.configure(state="disabled")
        self.save_button.configure(state="disabled")
        self.warning_var.set("Analisando…")
        self.progress.start(10)
        threading.Thread(target=self._analyze, daemon=True).start()

    def _analyze(self) -> None:
        try:
            classifier, freeze = _load_paths()
            if not classifier.is_file():
                raise FileNotFoundError(f"Modelo não encontrado:\n{classifier}")
            if not freeze.is_file():
                raise FileNotFoundError(f"Congelamento não encontrado:\n{freeze}")
            result = infer_fault(self.selected_file, classifier, freeze)
            self.after(0, self._show_result, result)
        except Exception as error:
            self.after(0, self._show_error, str(error))

    def _show_result(self, result: dict[str, object]) -> None:
        self.progress.stop()
        self.analyze_button.configure(state="normal")
        self.save_button.configure(state="normal")
        self.last_result = result
        classification = result["classification"]
        location = result["location"]
        code = classification["fault_class"]
        self.class_var.set(f"Tipo de falta: {code} — {CLASS_NAMES.get(code, code)}")
        self.vote_var.set(f"Concordância das árvores: {classification['tree_vote_fraction'] * 100:.1f}%")
        if location["conclusive"]:
            self.location_var.set(f"Distância desde PDT: {location['distance_from_PDT_km']:.2f} km")
            warning = result["location_domain_warning"]
        else:
            self.location_var.set("Distância desde PDT: resultado inconclusivo")
            warning = location.get("reason") or "Não foi possível localizar com segurança."
        self.snr_var.set(f"SNR pré-falta estimado: {result['estimated_prefault_snr_db']:.1f} dB")
        if classification["tree_vote_fraction"] < LOW_CONFIDENCE_THRESHOLD:
            warning = f"{NO_SIGNATURE_HINT}\n\n{warning}" if warning else NO_SIGNATURE_HINT
        self.warning_var.set(warning)

    def _show_error(self, message: str) -> None:
        self.progress.stop()
        self.analyze_button.configure(state="normal")
        self.warning_var.set("")
        messagebox.showerror("Erro na análise", message)

    def save_result(self) -> None:
        if self.last_result is None:
            return
        suggested = f"resultado_{self.selected_file.stem}.json" if self.selected_file else "resultado.json"
        target = filedialog.asksaveasfilename(title="Salvar resultado", initialfile=suggested, defaultextension=".json", filetypes=(("Arquivo JSON", "*.json"),))
        if target:
            Path(target).write_text(json.dumps(self.last_result, ensure_ascii=False, indent=2), encoding="utf-8")
            messagebox.showinfo("Resultado salvo", f"Arquivo salvo em:\n{target}")


if __name__ == "__main__":
    ClassifierApp().mainloop()
