# Classificador e Localizador de Faltas em Linhas de Transmissão

Pipeline de **detecção, classificação e localização de faltas** em linhas de
transmissão de energia, a partir de simulações eletromagnéticas transitórias
(ATP/ATPDraw), usando extração de atributos físicos e *machine learning*.

Projeto de Iniciação Científica (PIBIC) — Universidade Federal do Piauí.

---

## Sumário

- [Visão geral](#visão-geral)
- [Resultados](#resultados)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Como usar (produto final)](#como-usar-produto-final)
- [Como usar (código-fonte)](#como-usar-código-fonte)
- [Como funciona](#como-funciona)
- [Faixas de parâmetros validadas](#faixas-de-parâmetros-validadas)
- [Limitações conhecidas](#limitações-conhecidas)
- [Testes automatizados](#testes-automatizados)
- [Histórico de versões do modelo](#histórico-de-versões-do-modelo)

---

## Visão geral

Dado um sinal de tensão e corrente de dois terminais de uma linha de
transmissão (arquivo `.pl4` gerado pelo ATP), o pipeline:

1. **Detecta** o instante do evento de falta a partir de um limiar adaptativo
   (mediana + desvio absoluto) sobre a energia do sinal, sem depender de um
   instante fixo pré-configurado.
2. **Classifica** o tipo de falta entre 10 classes (`AG`, `BG`, `CG`, `AB`,
   `BC`, `CA`, `ABG`, `BCG`, `CAG`, `ABC`) usando um ensemble de árvores
   (ExtraTrees) treinado sobre 61 atributos físicos (RMS, picos, componentes
   simétricas, degraus transitórios).
3. **Localiza** a distância da falta a partir do terminal local por
   correlação de ondas viajantes (frente incidente vs. reflexão), com
   verificação cruzada entre os dois terminais.

O produto final é um aplicativo gráfico standalone (`.exe`, não requer
Python instalado) que recebe um `.pl4` e devolve classe, distância,
confiança e a forma de onda do evento.

## Resultados

| Etapa | Métrica | Valor |
|---|---|---|
| Classificação — teste cego oficial (70 casos, 7 condições) | Macro-F1 / acurácia | 100% |
| Classificação — validação independente acumulada (300 casos) | Acurácia | 99,7% |
| Localização — teste cego oficial (condição ideal, 15-450 km) | MAE / mediana / P95 | 0,665 km / 0,358 km / 2,834 km |
| Localização — revalidação informal 500-600 km (após correção da janela de normalização) | MAE / máximo | 1,66 km / 7,12 km |
| Validação em escala (280 casos, toda a faixa 15-600 km/Rfault/ângulo/t_cl) | Classificação / localização | 99,64% / 96,1% conclusivo, MAE 2,61 km, **1,5% falso-conclusivo entre os conclusivos** |
| Confiança média das árvores (após calibração Platt/sigmoid) | — | ~88,7% |

Todos os números de validação independente vêm de lotes gerados **depois**
do treino, nunca reaproveitados como dado de treino (auditoria automática de
vazamento em `src/manifest.py`).

## Estrutura do projeto

```
app/                      Produto final para uso direto
├── ClassificadorFaltasATP.exe   Aplicativo standalone (não precisa de Python)
├── classificador_config.json    Aponta para o modelo ativo em modelos/
└── ABRIR_CLASSIFICADOR.bat      Atalho de execução

src/                       Código-fonte Python (pipeline, treino, GUI)
├── classificador_gui_v2.py       Interface gráfica (com gráfico da forma de onda)
├── infer_fault.py                Inferência via linha de comando
├── feature_extraction.py         Extração dos 61 atributos físicos
├── pl4_reader.py / signal_io.py  Leitura de arquivos .pl4 / .adf
├── traveling_wave_localizer.py   Localização por correlação de ondas viajantes
├── adaptive_localizer.py         Localização multi-banda com verificação de SNR
├── fault_case_generator.py       Geração de casos de falta no ATP
├── simulation_generator.py       Geração e execução de simulações ATP
├── manifest.py                    Validação e auditoria anti-vazamento do manifesto
├── train_*.py                     Scripts de treino dos classificadores
└── ABRIR_EM_MODO_DESENVOLVEDOR.bat  Roda a GUI a partir do código-fonte (requer Python)

modelos/                   Artefatos congelados
├── robust_classifier_v5_calibrated.joblib   Classificador ativo
├── distance_sanity_regressor.joblib         Checagem de sanidade do localizador (ML)
├── FINAL_PIPELINE_FREEZE_V12.json           Congelamento ativo + histórico completo
└── classificador_config.master.json

docs/                       Relatórios e documentação de continuidade
resultados_experimentos/    Saídas de experimentos (curva de aprendizado, comparação de modelos)
legado/                      Abordagem anterior (wavelet/limiar), mantida como referência histórica
tests/                       Suíte de testes automatizados
run_tests.py                 Executa toda a suíte (python run_tests.py)
```

## Como usar (produto final)

1. Dê dois cliques em `app/ABRIR_CLASSIFICADOR.bat` (ou direto em
   `ClassificadorFaltasATP.exe`).
2. Clique em **Escolher PL4…** e selecione o arquivo gerado pelo ATPDraw.
3. Clique em **Analisar**.
4. Veja o tipo de falta, a distância, a confiança e a forma de onda no
   gráfico.
5. Use **Salvar resultado (JSON)…** para guardar o resultado.

Não é necessário ter Python instalado — o `.exe` é standalone.

## Como usar (código-fonte)

Requer Python 3.11+ e as dependências em uso pelo projeto (`scikit-learn`,
`numpy`, `scipy`, `joblib`, `matplotlib`).

```powershell
cd src
python classificador_gui_v2.py
```

Ou, para inferência direta em linha de comando:

```powershell
cd src
python infer_fault.py "C:\caminho\novo.pl4" `
  --classifier "..\modelos\robust_classifier_v5_calibrated.joblib" `
  --freeze "..\modelos\FINAL_PIPELINE_FREEZE_V12.json" `
  --output resultado.json
```

O resultado JSON informa a classe prevista, a fração de votos das árvores
(não é uma probabilidade calibrada por si só — a calibração já foi aplicada
sobre o classificador), o SNR pré-falta estimado, e a distância desde o
terminal local. A localização não é divulgada fora da faixa validada.

## Como funciona

### Extração de atributos (`feature_extraction.py`)

O instante do evento é localizado por comparação ciclo-a-ciclo (diferença
entre a amostra atual e a amostra um ciclo de 60 Hz antes), suavizada e
comparada a um limiar adaptativo derivado da mediana e do desvio absoluto
mediano (MAD) do próprio sinal pré-falta — não um valor fixo arbitrário.
A partir desse instante são extraídos 61 atributos: razões RMS e de pico
antes/depois do evento, degrau transitório máximo, e razões de componentes
simétricas (sequência zero/positiva/negativa) de tensão e corrente em
ambos os terminais.

### Classificação

Um `ExtraTreesClassifier` (scikit-learn) é treinado sobre milhares de casos
simulados em 10 classes de falta, selecionado entre vários candidatos
(diferentes tamanhos de folha, diferentes algoritmos — comparado também
contra RandomForest, Gradient Boosting e redes neurais MLP) pelo pior caso
de F1-macro em 7 condições de robustez (ruído, erro de ganho, erro de
sincronização). A confiança das árvores é recalibrada com
`CalibratedClassifierCV` (método sigmoid/Platt) para refletir melhor a
frequência real de acerto.

### Localização

A distância é obtida por correlação entre a frente de onda incidente e sua
reflexão, nos componentes aéreos de Clarke (transformação modal alfa/beta),
com verificação cruzada de consistência entre os dois terminais. Abaixo de
um SNR estimado de 50 dB, ou fora da janela temporal em que o método foi
validado, a localização é bloqueada e reportada como inconclusiva — a
classificação permanece disponível de qualquer forma, pois é o componente
validado com maior robustez.

## Faixas de parâmetros validadas

| Parâmetro | Faixa | Justificativa |
|---|---|---|
| Classificação (10 classes) | 15 – 600 km | limite físico usual de linha CA sem compensação série |
| Localização | 15 – 600 km | teste cego oficial cobriu 15-450 km; 450-600 km foi revalidado informalmente após a correção da janela de normalização (v9) |
| Resistência de falta (Rfault) | 0,01 – 3000 Ω | de curto franco até falta de alta impedância (vegetação/solo seco) |
| Ângulo de incidência | 0° – 360° | a falta pode ocorrer em qualquer ponto do ciclo de 60 Hz |
| Instante de fechamento (t_cl) — classificação e localização | 0,025 – 0,675 s | livre dentro da simulação (Tmax=0,7s no template ATP); antes da v9 o localizador exigia 0,080-0,105 s por um bug de normalização, já corrigido; janela ampliada em v11 e v12. Tmax não pode subir mais sem revalidar — acima de ~800 mil passos de tempo (Tmax≈0,8-0,9s) o solver ATP corrompe os resultados |

**Importante para uso manual no ATPDraw**: a ampliação de t_cl depende do
`Tmax` da simulação ser 0,7s. Isso já está configurado no template usado
pelos scripts Python (`simulation_generator.py`), mas se você gerar o
caso manualmente pelo ATPDraw (fora do pipeline Python), o `Tmax` do seu
projeto também precisa ser ajustado para 0,7s — caso contrário a
simulação continua limitada ao Tmax antigo e um t_cl acima de ~100ms pode
não caber na janela de observação pós-falta.

**Não aumente o Tmax além de 0,7s sem revalidar**: testamos até 1,0s e
descobrimos que o solver ATP (`tpbig.exe`) corrompe silenciosamente os
resultados acima de ~800-900 mil passos de tempo (Tmax≈0,8-0,9s com o
passo de 1µs usado aqui) — não é degradação gradual, é uma quebra abrupta
de precisão. 0,7s foi escolhido com margem de segurança abaixo desse
limite.

`ABC-G` (trifásica-terra) não é uma classe suportada: testes de viabilidade
mostraram que ela não é separável de `ABC` com confiabilidade estatística,
o que é consistente com a física do problema (uma falta trifásica simétrica
produz corrente de neutro próxima de zero, com ou sem aterramento).

## Limitações conhecidas

- **Risco residual do localizador (reduzido, não eliminado)**: mesmo após
  a correção da janela de normalização, uma bateria de 280 casos mostrou
  que ~1,5% dos casos conclusivos ainda recebiam uma distância errada com
  falsa confiança (erro de 58-162 km). Três filtros baseados no próprio
  sinal de reflexão (margem entre candidatos, consenso multiescala,
  consistência entre terminais) não resolveram isso — nos casos ruins, os
  dois terminais concordam consistentemente na mesma reflexão errada
  (provável segundo salto), então não é ruído de medição. A solução foi
  adicionar uma **checagem de sanidade por machine learning**: um
  `RandomForestRegressor` estima a distância pela atenuação do sinal (um
  princípio físico independente da correlação de reflexão); quando as
  duas estimativas discordam por mais de 100 km, o resultado vira
  inconclusivo em vez de reportado com confiança falsa. Isso reduziu a
  taxa de falso-conclusivo de 1,49% para 0,38% dos casos conclusivos, ao
  custo de perder ~2% de cobertura (casos corretos ocasionalmente
  rejeitados à toa). O risco não foi eliminado — trate a distância
  reportada como estimativa, não como garantia absoluta.
- Há uma confusão residual muito pontual entre `ABG` e `AB` em distâncias
  muito curtas (1 caso em 300 testados na v5); não foi reproduzida em 40
  casos novos testados na v9 — tratada como ruído estatístico, não um
  padrão sistemático corrigível.
- As validações incrementais (v2 em diante) são lotes informais pós-treino;
  apenas o teste cego oficial (v1) segue metodologia de campanha cega
  completa com splits bloqueados antes do treino.
- A faixa 600-620 km do localizador é apenas folga técnica no teto de busca
  e não foi validada com casos reais.

Duas limitações documentadas em versões anteriores (localização bloqueada
fora de 80-105 ms de fechamento de falta, e reflexões falsas aceitas acima
de 450 km) tinham a mesma causa raiz — uma janela de normalização presa ao
instante clássico de falta — corrigida na v9 (ver histórico abaixo).

## Testes automatizados

```powershell
python run_tests.py
```

## Histórico de versões do modelo

Resumo da evolução do classificador; detalhes completos em
`modelos/FINAL_PIPELINE_FREEZE_V8.json` e `docs/RELATORIO_FINAL.md`.

| Versão | Mudança | Resultado |
|---|---|---|
| v1 | Linha de base — 500 casos, teste cego oficial | Macro-F1 100% (7 condições); localizador MAE 0,665 km |
| v2 | +36 casos de faltas com terra em 200–450 km | 91,8% em validação independente |
| v3 | +80 casos em 450–600 km | 96,0% |
| v4 | +450 casos (ênfase 200–600 km) | 98,7% combinado |
| v5 | +80 casos sem terra em 15–80 km | 99,7% combinado |
| v6 | Calibração de confiança (Platt/sigmoid) | 99,7% mantido; confiança média 75% → 88,7% |
| v7 | Faixa de Rfault ampliada (100 Ω → 3000 Ω), validada sem retreino | 100% em faixa estendida |
| v8 | Janela de detecção do classificador ampliada; guarda de segurança adicionada ao localizador | 99,7% mantido com t_cl livre |
| v9 | Corrigido bug de normalização no localizador (janela de baseline fixa → relativa ao início da simulação) | t_cl fora de 80-105ms: erro caiu de 79-243km para 0,31km; 500-600km: 0 falso-conclusivos em 30 casos, MAE 1,66km |
| v9-batch | Validação em escala (280 casos, toda a faixa de parâmetros) | Classificação 99,64%; localização 96,1% conclusivo, MAE 2,61km, 1,5% falso-conclusivo entre os conclusivos (risco residual documentado) |
| v10 | Regressor de ML como checagem de sanidade independente do localizador (estima distância pela atenuação do sinal, não pela correlação de reflexão) | Falso-conclusivo cai de 1,49% para 0,38% (redução de ~4x); cobertura cai de 96,1% para 94,3% (2 casos bons rejeitados à toa); MAE dos corretos melhora para 0,99km |
| v11 | Tmax do template ATP ampliado de 0,15s para 0,5s; janela de t_cl ampliada de 25-125ms para 25-475ms; regressor de sanidade retreinado com casos cobrindo a faixa nova | 40 casos novos com t_cl uniforme em 25-475ms: classificação 100%, localização 92,5% conclusivo, 0 falso-conclusivos, MAE 0,99km |
| v12 | Descoberto limite numérico do solver ATP: acima de ~800-900 mil passos de tempo (Tmax≈0,8-0,9s) a precisão corrompe abruptamente. Tmax fixado em 0,7s (700 mil passos, com margem); janela de t_cl final 25-675ms; regressor de sanidade retreinado sem os dados contaminados por Tmax=1,0s | 40 casos novos com t_cl uniforme em 25-675ms: classificação 100%, localização 97,5% conclusivo, 1 falso-conclusivo (2,5%, dentro do risco residual já conhecido), MAE 3,16km |

---

Desenvolvimento de pesquisa acadêmica (PIBIC) — Universidade Federal do Piauí.
