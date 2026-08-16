# Experimentos históricos

Scripts de versões e experimentos anteriores do classificador, mantidos
como registro do processo de pesquisa (o que foi tentado, e por quê),
mas **não fazem parte do pipeline em produção**. Para o código ativo, veja
`src/` (raiz) e o [README principal](../../README.md).

- `calibracao/` — scripts de calibração de confiança de cada versão do
  classificador (v13 até v31, mais os experimentos pontuais de feature
  `zero_over_positive_current` e `modal`). O script ativo hoje é
  `src/calibrate_modal_final.py`.
- `treino/` — scripts de treino de cada versão do classificador, incluindo
  abordagens alternativas descartadas (baseline simples, classificador
  hierárquico). O script ativo hoje é `src/train_robust_classifier_v31.py`.
- `analises/` — avaliações pontuais feitas durante a pesquisa: viabilidade
  da classe `ABC-G` (descartada — ver `modelos/FINAL_PIPELINE_FREEZE_G2.2.json`),
  testes de alta impedância, comparação com MLP/outros modelos, extensão de
  faixa de distância, curva de aprendizado, e a interface gráfica v1
  (substituída por `classificador_gui_v2.py`).

O motivo e o resultado de cada mudança de versão está documentado nos
arquivos `modelos/historico/FINAL_PIPELINE_FREEZE_*.json` e
`modelos/FINAL_PIPELINE_FREEZE_G2.*.json` — cada um referencia o anterior
em `based_on`, formando o histórico completo até a versão em produção.
