# Continuação do projeto por outro agente

Atualizado após a avaliação final em 3 de agosto de 2026. Este arquivo deve permitir que outro agente
continue o trabalho sem reconstruir decisões já tomadas.

## Objetivo

Criar uma ferramenta que leia um PL4 ATPDraw com medições sincronizadas em PDT
e BEA, classifique uma falta e estime sua distância em quilômetros a partir de
PDT. Toda entrada contém uma falta.

## Projeto e dados

- Código: raiz do repositório
- Campanha: `pilot\campaign_v4` (diretório local de trabalho, fora do repositório)
- Manifesto: `campaign_v4\manifest.csv`
- Plano bloqueado: `pilot\pilot_plan_10classes.csv`
- ATP autorizado: `C:\ATP\atpmingw\tpbig.exe`
- Template original: `C:\RESULTPESQUISA\SIMULACAOUSADA.atp`

## Requisitos congelados

- 10 classes: `AG`, `BG`, `CG`, `AB`, `BC`, `CA`, `ABG`, `BCG`, `CAG`, `ABC`.
- `ABC` e `ABCG` foram consolidadas porque são praticamente indistinguíveis no sistema equilibrado.
- Frequência 60 Hz, passo 1 µs, Tmax 150 ms, falta após cinco ciclos mais ângulo.
- Distância desde PDT: 1–500 km; trecho remoto até 500 km; linha total >=100 km.
- Rfault 0,01–100 ohms e participa fisicamente de todas as classes.
- Linha JMarti nas duas seções; 12 canais: VA/VB/VC/IA/IB/IC em PDT e BEA.
- O modelo nunca pode usar nome/caminho do arquivo, run_id, classe ou parâmetros físicos como entrada.

## Campanha concluída

- 50 cenários físicos pareados nas 10 classes: 500 PL4, aproximadamente 3,97 GB.
- Divisão bloqueada: 350 treino, 80 validação, 70 `test_unseen`.
- Todos os PL4 foram reabertos e validados: 12 canais, 150.001 amostras, valores finitos.
- Plano SHA-256: `60111d22abfe9fd32f0f7a68c8919974d368368f79650009c671ce8abd80f931`.
- Manifesto SHA-256: `61939156e98ce96c2b69fbf0f28f901c490a2a1a7bd2f83d84e8d2cf722cf9d7`.

## Código principal

- `pl4_reader.py`: leitor NEWPL4=2.
- `signal_io.py`: perfil canônico dos 12 canais.
- `fault_case_generator.py`: redes de falta resistivas corretas e chaves permanentes até após Tmax.
- `jmarti_generator.py`, `simulation_generator.py`, `pilot_runner.py`: geração reproduzível.
- `feature_extraction.py`: detecção automática e 61 atributos; limiar mínimo atual 0,01.
- `train_baseline.py`: classificador ExtraTrees e regressor-base.
- `traveling_wave_localizer.py`: localizador por primeiras frentes e reflexões direcionais.
- `evaluate_traveling_wave.py`: calibração no treino e avaliação na validação.
- `robustness_evaluation.py`: perturbações determinísticas em memória.
- `classificador_gui.py` e `ABRIR_CLASSIFICADOR.bat`: interface gráfica de dois cliques.
- `classificador_config.json`: caminho do classificador usado pela interface.

## Resultados atuais

Classificador treinado somente nos 350 casos de treino:

- ideal, ganho ±1%, sincronização ±1 µs e SNR 60 dB: macro-F1 100%;
- SNR 40 dB: macro-F1 85,50%;
- SNR 30 dB: macro-F1 64,52%;
- combinado 30 dB+ganho+sincronização: macro-F1 61,99%.

Localizador ideal, considerando somente respostas conclusivas:

- cobertura 98,75%; MAE 0,543 km; mediana 0,355 km; P95 2,244 km; máximo 3,413 km.
- velocidade calibrada somente no treino: 0,2992746507 km/µs.
- correlação mínima 0,30 em ambos os terminais marca resultado conclusivo.

Sob ruído, o localizador ainda falha: em 40 dB a cobertura cai para 61,25% e
há uma falsa reflexão conclusiva; em 30 dB ele não é confiável.

Relatórios:

- `campaign_v4\baseline_v2\validation_report.json`
- `campaign_v4\traveling_wave_v1\validation_report.json`
- `campaign_v4\robustness_v1\validation_report.json`

## Estado científico do teste cego

Os 70 casos `test_unseen` foram abertos uma única vez depois do congelamento
`FINAL_PIPELINE_FREEZE.json` (SHA-256
`7EC8C7B41E2C5B251BBC43ADA6A365019865F58C07F3CE3FA299166454E670D5`).
O relatório final é `campaign_v4\final_blind_test\report.json`, SHA-256
`20BDEC7CD6E08FDD57CEC088BFF23076EFF8AEC03DACD9B37A2E9DD0909F3771`.
Esses casos não podem mais ser usados para ajustes. Qualquer novo ajuste que
pretenda métricas cegas exige uma nova campanha `test_unseen`.

## Resultado final e trabalho futuro

O classificador final atingiu macro-F1 `100%` nas sete condições cegas.
Localização cega ideal: cobertura `100%`, MAE `0,665 km`, mediana `0,358 km`,
P95 `2,834 km`, máximo `4,368 km`. Ganho ±1% e sync ±1 µs também passaram.
Em 60 dB: cobertura `92,86%`, MAE `0,702 km`, P95 `2,960 km`, máximo `4,218 km`.
Em 40 dB e no combinado de 30 dB houve falsas reflexões conclusivas e as metas
não foram atendidas. Por segurança, `infer_fault.py` não divulga distância
quando o SNR pré-falta estimado é inferior a 50 dB; essa guarda é posterior ao
teste e precisa de uma nova campanha cega para receber uma métrica própria.
O teste cego cobriu distâncias de 15 a 450 km. Um smoke test posterior no caso
de treino a 1 km estimou 15,84 km; portanto a localização próxima ao terminal
não está resolvida e a faixa 1–500 km não pode ser alegada como validada.

## Regras de trabalho

- Preservar artefatos existentes; criar versões novas (`baseline_v3`, `robustness_v2`, etc.).
- Não misturar PCH antigo e novo.
- Usar `apply_patch` para editar arquivos.
- Executar `python -m unittest discover -s tests -p "test_*.py" -v` após mudanças.
- Atualizar este arquivo, `CONTEXTO_PROJETO.md` e `README.md` com resultados reais.
- Relatar falhas honestamente; nunca promover uma métrica excluindo erros sem declarar cobertura/inconclusivos.
