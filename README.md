# Classificador e Localizador de Faltas — PIBIC UFPI

Ferramentas para classificação de tipo de falta e localização de distância
em linhas de transmissão, a partir de simulações ATP/ATPDraw, usando
extração de atributos físicos + machine learning (ExtraTrees/RandomForest).

## Estrutura do projeto

```
app/                    Produto final para uso — dê 2 cliques em ABRIR_CLASSIFICADOR.bat
├── ClassificadorFaltasATP.exe   Aplicativo standalone (não precisa de Python instalado)
├── classificador_config.json    Aponta para o modelo ativo (modelos/)
└── ABRIR_CLASSIFICADOR.bat

src/                     Todo o código-fonte Python (pipeline, treino, GUI)
├── classificador_gui_v2.py       Interface gráfica (com gráfico da forma de onda)
├── infer_fault.py                Inferência via linha de comando
├── feature_extraction.py         Extração dos 61 atributos físicos
├── pl4_reader.py / signal_io.py  Leitura de .pl4 / .adf
├── *_localizer.py                Localização por ondas viajantes
├── fault_case_generator.py       Geração de casos de falta no ATP
├── simulation_generator.py       Geração + execução de simulações ATP
├── train_*.py                    Scripts de treino dos classificadores
└── ABRIR_EM_MODO_DESENVOLVEDOR.bat  Roda a GUI a partir do código-fonte (requer Python)

modelos/                 Artefatos congelados (classificador treinado + documentação)
├── robust_classifier_v5.joblib
├── FINAL_PIPELINE_FREEZE_V5.json (e versões anteriores v1-v4, histórico)
└── classificador_config.master.json

docs/                    Relatórios e documentação do projeto
resultados_experimentos/ Saídas de experimentos (curva de aprendizado, comparação de modelos)
legado/                  Código e arquivos antigos, não usados no pipeline atual
tests/                   Suíte de testes automatizados (35 testes)
run_tests.py             Roda a suíte de testes (python run_tests.py)
```

## Uso rápido (produto final)

Dê dois cliques em `app\ABRIR_CLASSIFICADOR.bat` (ou direto em
`ClassificadorFaltasATP.exe`). Na janela:

1. clique em **Escolher PL4…**;
2. selecione o arquivo `.pl4` gerado pelo ATPDraw;
3. clique em **Analisar**;
4. veja o tipo de falta, distância, confiança e a forma de onda no gráfico;
5. use **Salvar resultado (JSON)…** se quiser guardar o resultado.

Não precisa ter Python instalado — o `.exe` é standalone.

## Uso via código-fonte / linha de comando

```powershell
cd src
python classificador_gui_v2.py

# ou, para inferência direta em linha de comando:
python infer_fault.py "C:\caminho\novo.pl4" `
  --classifier "..\modelos\robust_classifier_v5.joblib" `
  --freeze "..\modelos\FINAL_PIPELINE_FREEZE_V5.json" `
  --output resultado.json
```

O resultado informa classe, fração de votos das árvores (não é probabilidade
calibrada), SNR pré-falta estimado e distância desde PDT. A localização não é
divulgada fora da faixa validada (15-450 km) por segurança.

## Rodar os testes

```powershell
python run_tests.py
```

## Faixas de parâmetros validadas

| Parâmetro | Faixa | Justificativa |
|---|---|---|
| Classificação (10 classes: AG/BG/CG/AB/BC/CA/ABG/BCG/CAG/ABC) | 15 – 600 km | limite físico de linha CA sem compensação série |
| Localização | 15 – 450 km | acima disso o localizador aceita reflexão falsa como confiável |
| Rfault | 0,01 – 3000 Ω | de curto franco até falta de alta impedância (vegetação/solo seco) |
| Ângulo de incidência | 0° – 360° | falta pode ocorrer em qualquer ponto do ciclo de 60Hz |
| t_cl (fechamento da falta) — classificação | 0,025 – 0,125 s | qualquer instante dentro da simulação de 0,15s, com folga de regime permanente antes e janela pós-falta depois |
| t_cl (fechamento da falta) — localização | 0,080 – 0,105 s | o localizador ainda usa uma janela de referência fixa; fora dela, `infer_fault.py` bloqueia a distância em vez de arriscar um valor errado |

`ABC-G` (trifásica-terra) não é uma classe suportada — testes de viabilidade
mostraram que não é separável de `ABC` com confiabilidade (ver
`docs/RELATORIO_FINAL.md`).

Detalhes completos, histórico de versões (v1→v7) e limitações conhecidas em
`modelos/FINAL_PIPELINE_FREEZE_V7.json` e `docs/RELATORIO_FINAL.md`.

Desenvolvimento de pesquisa acadêmica pela Universidade Federal do Piauí (PIBIC).
