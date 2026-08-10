"""Interface grafica profissional do Classificador de Faltas ATP.

Mostra classificacao, localizacao e o grafico das tensoes trifasicas em
torno do instante da falta, alem de metadados do modelo ativo.
"""

from __future__ import annotations

import json
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from infer_fault import infer_fault
from feature_extraction import extract_features
from signal_io import read_canonical_pl4


# Em modo .exe (PyInstaller), __file__ aponta para a pasta temporaria de
# extracao (sys._MEIPASS), nao para onde o .exe realmente esta. Usamos a
# pasta do executavel nesse caso, para achar classificador_config.json ao
# lado do .exe.
if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).resolve().parent
else:
    BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "classificador_config.json"

CLASS_NAMES = {
    "AG": "Fase A – Terra", "BG": "Fase B – Terra", "CG": "Fase C – Terra",
    "AB": "Fases A – B", "BC": "Fases B – C", "CA": "Fases C – A",
    "ABG": "Fases A – B – Terra", "BCG": "Fases B – C – Terra", "CAG": "Fases C – A – Terra",
    "ABC": "Falta Trifásica",
}

LOW_CONFIDENCE_THRESHOLD = 0.30
NO_SIGNATURE_HINT = (
    "Confiança muito baixa: o sinal pode não conter uma falta real nesta janela "
    "(confira, no ATPDraw, se todas as chaves de falta fecham dentro de t_cl entre "
    "0,0833 e 0,1000 s, com T-op = 2)."
)

# Paleta
COLOR_BG = "#0f1626"
COLOR_PANEL = "#161f36"
COLOR_PANEL_ALT = "#1c2742"
COLOR_ACCENT = "#4da3ff"
COLOR_TEXT = "#e8ecf5"
COLOR_MUTED = "#8b96b3"
COLOR_OK = "#3ddc97"
COLOR_WARN = "#ffb84d"
COLOR_BAD = "#ff6b6b"
COLOR_PHASE_A = "#ffcc4d"
COLOR_PHASE_B = "#4da3ff"
COLOR_PHASE_C = "#ff6b9d"


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
    parts = [f"{freeze_path.name}  ·  modelo {version}…"]
    if classifier_range:
        parts.append(f"classificação {classifier_range[0]}-{classifier_range[1]}km")
    if localizer_range:
        parts.append(f"localização {localizer_range[0]}-{localizer_range[1]}km")
    return "   ·   ".join(parts)


def _confidence_color(fraction: float) -> str:
    if fraction >= 0.6:
        return COLOR_OK
    if fraction >= 0.3:
        return COLOR_WARN
    return COLOR_BAD


class Card(tk.Frame):
    def __init__(self, master: tk.Widget, **kwargs) -> None:
        super().__init__(master, bg=COLOR_PANEL, highlightthickness=1,
                          highlightbackground="#26325a", **kwargs)


class ClassifierApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Classificador de Faltas ATP")
        self.geometry("1180x760")
        self.minsize(1020, 680)
        self.configure(bg=COLOR_BG)
        self.selected_file: Path | None = None
        self.last_result: dict[str, object] | None = None
        self._build_style()
        self._build()

    def _build_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=COLOR_BG)
        style.configure("Panel.TFrame", background=COLOR_PANEL)
        style.configure("TLabel", background=COLOR_BG, foreground=COLOR_TEXT, font=("Segoe UI", 10))
        style.configure("Panel.TLabel", background=COLOR_PANEL, foreground=COLOR_TEXT, font=("Segoe UI", 10))
        style.configure("Muted.TLabel", background=COLOR_BG, foreground=COLOR_MUTED, font=("Segoe UI", 9))
        style.configure("PanelMuted.TLabel", background=COLOR_PANEL, foreground=COLOR_MUTED, font=("Segoe UI", 9))
        style.configure("Title.TLabel", background=COLOR_BG, foreground=COLOR_TEXT, font=("Segoe UI", 22, "bold"))
        style.configure("ResultTitle.TLabel", background=COLOR_PANEL, foreground=COLOR_TEXT, font=("Segoe UI", 20, "bold"))
        style.configure(
            "Accent.TButton", background=COLOR_ACCENT, foreground="#08101f",
            font=("Segoe UI", 10, "bold"), padding=(14, 8), borderwidth=0,
        )
        style.map("Accent.TButton", background=[("active", "#6bb6ff"), ("disabled", "#3a4666")])
        style.configure(
            "Secondary.TButton", background=COLOR_PANEL_ALT, foreground=COLOR_TEXT,
            font=("Segoe UI", 10), padding=(12, 7), borderwidth=0,
        )
        style.map("Secondary.TButton", background=[("active", "#28345a"), ("disabled", "#1a2338")])
        style.configure("TEntry", fieldbackground=COLOR_PANEL_ALT, foreground=COLOR_TEXT,
                        insertcolor=COLOR_TEXT, borderwidth=0)
        style.configure("TProgressbar", background=COLOR_ACCENT, troughcolor=COLOR_PANEL_ALT, borderwidth=0)

    def _build(self) -> None:
        outer = ttk.Frame(self, padding=24)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer)
        header.pack(fill="x")
        ttk.Label(header, text="⚡ Classificador de Faltas ATP", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header, text="Classificação de tipo de falta e localização em linhas de transmissão via ondas viajantes",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(2, 0))
        self.freeze_var = tk.StringVar(value="Carregando configuração…")
        ttk.Label(header, textvariable=self.freeze_var, style="Muted.TLabel").pack(anchor="w", pady=(6, 0))
        self._refresh_freeze_summary()

        chooser = ttk.Frame(outer)
        chooser.pack(fill="x", pady=(18, 0))
        self.file_var = tk.StringVar(value="Nenhum arquivo selecionado")
        entry = ttk.Entry(chooser, textvariable=self.file_var, state="readonly")
        entry.pack(side="left", fill="x", expand=True, ipady=5)
        ttk.Button(chooser, text="Escolher PL4…", style="Secondary.TButton", command=self.choose_file).pack(side="left", padx=(10, 0))
        self.analyze_button = ttk.Button(chooser, text="Analisar", style="Accent.TButton", command=self.start_analysis, state="disabled")
        self.analyze_button.pack(side="left", padx=(10, 0))

        self.progress = ttk.Progressbar(outer, mode="indeterminate", style="TProgressbar")
        self.progress.pack(fill="x", pady=(10, 0))

        body = ttk.Frame(outer)
        body.pack(fill="both", expand=True, pady=(18, 0))
        body.columnconfigure(0, weight=0, minsize=340)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        self._build_result_panel(body)
        self._build_plot_panel(body)

    def _build_result_panel(self, parent: ttk.Frame) -> None:
        card = Card(parent, padx=20, pady=20)
        card.grid(row=0, column=0, sticky="nsew", padx=(0, 16))

        tk.Label(card, text="RESULTADO", bg=COLOR_PANEL, fg=COLOR_MUTED, font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.class_label = tk.Label(card, text="—", bg=COLOR_PANEL, fg=COLOR_TEXT, font=("Segoe UI", 22, "bold"), wraplength=300, justify="left")
        self.class_label.pack(anchor="w", pady=(4, 0))
        self.class_desc_var = tk.StringVar(value="Selecione um arquivo PL4 para começar.")
        ttk.Label(card, textvariable=self.class_desc_var, style="PanelMuted.TLabel", wraplength=300).pack(anchor="w", pady=(0, 14))

        self._build_confidence_bar(card)

        divider1 = tk.Frame(card, bg="#26325a", height=1)
        divider1.pack(fill="x", pady=14)

        tk.Label(card, text="LOCALIZAÇÃO", bg=COLOR_PANEL, fg=COLOR_MUTED, font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.location_var = tk.StringVar(value="—")
        tk.Label(card, textvariable=self.location_var, bg=COLOR_PANEL, fg=COLOR_TEXT, font=("Segoe UI", 15, "bold")).pack(anchor="w", pady=(4, 12))

        tk.Label(card, text="QUALIDADE DO SINAL", bg=COLOR_PANEL, fg=COLOR_MUTED, font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.snr_var = tk.StringVar(value="—")
        ttk.Label(card, textvariable=self.snr_var, style="Panel.TLabel").pack(anchor="w", pady=(4, 0))

        divider2 = tk.Frame(card, bg="#26325a", height=1)
        divider2.pack(fill="x", pady=14)

        self.warning_var = tk.StringVar(value="")
        self.warning_label = tk.Label(
            card, textvariable=self.warning_var, bg=COLOR_PANEL, fg=COLOR_WARN,
            font=("Segoe UI", 9), wraplength=300, justify="left",
        )
        self.warning_label.pack(anchor="w")

        spacer = tk.Frame(card, bg=COLOR_PANEL)
        spacer.pack(fill="both", expand=True)

        self.save_button = ttk.Button(card, text="Salvar resultado (JSON)…", style="Secondary.TButton", command=self.save_result, state="disabled")
        self.save_button.pack(anchor="w", fill="x", pady=(10, 0))

    def _build_confidence_bar(self, card: tk.Widget) -> None:
        row = tk.Frame(card, bg=COLOR_PANEL)
        row.pack(fill="x")
        tk.Label(row, text="Confiança:", bg=COLOR_PANEL, fg=COLOR_MUTED, font=("Segoe UI", 9)).pack(side="left")
        self.confidence_pct_var = tk.StringVar(value="—")
        tk.Label(row, textvariable=self.confidence_pct_var, bg=COLOR_PANEL, fg=COLOR_TEXT, font=("Segoe UI", 9, "bold")).pack(side="right")
        track = tk.Frame(card, bg=COLOR_PANEL_ALT, height=8)
        track.pack(fill="x", pady=(4, 0))
        self.confidence_fill = tk.Frame(track, bg=COLOR_ACCENT, height=8, width=0)
        self.confidence_fill.place(x=0, y=0, relheight=1)
        self._confidence_track = track

    def _build_plot_panel(self, parent: ttk.Frame) -> None:
        card = Card(parent, padx=16, pady=16)
        card.grid(row=0, column=1, sticky="nsew")
        tk.Label(card, text="FORMA DE ONDA (TENSÃO PDT)", bg=COLOR_PANEL, fg=COLOR_MUTED, font=("Segoe UI", 9, "bold")).pack(anchor="w")

        self.figure = Figure(figsize=(6, 5), dpi=100, facecolor=COLOR_PANEL)
        self.ax = self.figure.add_subplot(111)
        self._style_axes()
        self.canvas = FigureCanvasTkAgg(self.figure, master=card)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, pady=(8, 0))
        self._plot_placeholder()

    def _style_axes(self) -> None:
        self.ax.set_facecolor(COLOR_PANEL)
        for spine in self.ax.spines.values():
            spine.set_color("#3a4666")
        self.ax.tick_params(colors=COLOR_MUTED, labelsize=8)
        self.ax.xaxis.label.set_color(COLOR_MUTED)
        self.ax.yaxis.label.set_color(COLOR_MUTED)
        self.ax.grid(True, color="#26325a", linewidth=0.6, alpha=0.6)

    def _plot_placeholder(self) -> None:
        self.ax.clear()
        self._style_axes()
        self.ax.text(
            0.5, 0.5, "Selecione e analise um arquivo PL4\npara ver a forma de onda",
            transform=self.ax.transAxes, ha="center", va="center", color=COLOR_MUTED, fontsize=11,
        )
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas.draw_idle()

    def _plot_waveform(self, pl4_path: Path) -> None:
        try:
            signals = read_canonical_pl4(pl4_path)
            features = extract_features(signals)
        except Exception:
            return
        t = signals.time_s
        va, vb, vc = signals.values[:, 0], signals.values[:, 1], signals.values[:, 2]
        event = features.event_time_s
        lo = max(float(t[0]), event - 0.010)
        hi = min(float(t[-1]), event + 0.030)
        mask = (t >= lo) & (t <= hi)

        self.ax.clear()
        self._style_axes()
        self.ax.plot(t[mask] * 1000, va[mask], color=COLOR_PHASE_A, linewidth=1.1, label="Fase A")
        self.ax.plot(t[mask] * 1000, vb[mask], color=COLOR_PHASE_B, linewidth=1.1, label="Fase B")
        self.ax.plot(t[mask] * 1000, vc[mask], color=COLOR_PHASE_C, linewidth=1.1, label="Fase C")
        self.ax.axvline(event * 1000, color=COLOR_TEXT, linestyle="--", linewidth=1, alpha=0.7, label="Instante da falta")
        self.ax.set_xlabel("Tempo (ms)", fontsize=9)
        self.ax.set_ylabel("Tensão PDT (V)", fontsize=9)
        legend = self.ax.legend(loc="upper right", fontsize=8, facecolor=COLOR_PANEL_ALT, edgecolor="#3a4666", labelcolor=COLOR_TEXT)
        self.figure.tight_layout()
        self.canvas.draw_idle()

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
        vote = classification["tree_vote_fraction"]

        self.class_label.configure(text=code)
        self.class_desc_var.set(CLASS_NAMES.get(code, code))

        color = _confidence_color(vote)
        self.confidence_pct_var.set(f"{vote * 100:.1f}%")
        self.confidence_fill.configure(bg=color)
        self.update_idletasks()
        track_width = self._confidence_track.winfo_width() or 300
        self.confidence_fill.place(x=0, y=0, relheight=1, width=max(4, int(track_width * min(vote, 1.0))))

        if location["conclusive"]:
            self.location_var.set(f"{location['distance_from_PDT_km']:.2f} km desde PDT")
            warning = result["location_domain_warning"]
        else:
            self.location_var.set("Inconclusiva")
            warning = location.get("reason") or "Não foi possível localizar com segurança."
        self.snr_var.set(f"{result['estimated_prefault_snr_db']:.1f} dB (SNR pré-falta estimado)")

        if vote < LOW_CONFIDENCE_THRESHOLD:
            warning = f"{NO_SIGNATURE_HINT}\n\n{warning}" if warning else NO_SIGNATURE_HINT
            self.warning_label.configure(fg=COLOR_BAD)
        else:
            self.warning_label.configure(fg=COLOR_WARN)
        self.warning_var.set(warning)

        if self.selected_file is not None:
            self._plot_waveform(self.selected_file)

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
