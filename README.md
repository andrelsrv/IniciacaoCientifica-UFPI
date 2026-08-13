<div align="center">

# ⚡ Classificador e Localizador de Faltas em Linhas de Transmissão

**Detecção, classificação e localização de faltas elétricas** a partir de
simulações eletromagnéticas transitórias (ATP/ATPDraw), usando extração de
atributos físicos e *machine learning*.

Projeto de Iniciação Científica (PIBIC) — Universidade Federal do Piauí

[![Download](https://img.shields.io/badge/⬇%20Download-Vers%C3%A3o%20portátil%20(.zip)-2ea44f?style=for-the-badge)](https://github.com/andrelsrv/IC-UFPI/releases/latest/download/ClassificadorFaltasATP-v1.0-portable.zip)
[![Releases](https://img.shields.io/github/v/release/andrelsrv/IC-UFPI?style=for-the-badge&label=vers%C3%A3o)](https://github.com/andrelsrv/IC-UFPI/releases)
[![Testes](https://img.shields.io/badge/testes-35%2F35%20passando-brightgreen?style=for-the-badge)](#testes-automatizados)
[![Licença](https://img.shields.io/badge/licença-MIT-blue?style=for-the-badge)](LICENSE)

</div>

---

## Sumário

- [Visão geral](#visão-geral)
- [Download rápido](#download-rápido)
- [Resultados](#resultados)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Como usar (produto final)](#como-usar-produto-final)
- [Como usar (código-fonte)](#como-usar-código-fonte)
- [Como funciona](#como-funciona)
- [Faixas de parâmetros validadas](#faixas-de-parâmetros-validadas)
- [Limitações conhecidas](#limitações-conhecidas)
- [Privacidade e segurança](#privacidade-e-segurança)
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
   verificação cruzada entre os dois terminais e uma checagem de sanidade
   independente por *machine learning*.

O produto final é um aplicativo gráfico standalone (`.exe`) que recebe um
`.pl4` e devolve classe, distância, confiança e a forma de onda do evento.

## Download rápido

<table>
<tr>
<td>

**[⬇ Baixar a versão portátil (.zip, ~145 MB)](https://github.com/andrelsrv/IC-UFPI/releases/latest/download/ClassificadorFaltasATP-v1.0-portable.zip)**

1. Baixe e extraia o `.zip` em qualquer pasta.
2. Abra a pasta `app/` e dê dois cliques em `ABRIR_CLASSIFICADOR.bat`.
3. Pronto — não precisa de Python nem de nenhuma biblioteca instalada.

</td>
</tr>
</table>

> **Por que um `.zip` separado, e não o botão "Code → Download ZIP" do
> GitHub?** O repositório usa Git LFS para os arquivos grandes (o `.exe` e
> os modelos treinados). O download padrão do GitHub **não baixa o
> conteúdo real desses arquivos** — só um ponteiro de texto — e o programa
> não funcionaria. O `.zip` da seção [Releases](https://github.com/andrelsrv/IC-UFPI/releases)
> já vem com tudo resolvido e pronto para uso.

Se você é desenvolvedor e quer o código-fonte com histórico completo, use
`git clone` (requer [Git LFS](https://git-lfs.com/) instalado) — veja
[Como usar (código-fonte)](#como-usar-código-fonte).

## Resultados

| Etapa | Métrica | Valor |
|---|---|---|
| Classificação — teste cego oficial (70 casos, 7 condições) | Macro-F1 / acurácia | **100%** |
| Classificação — validação independente acumulada (300+ casos) | Acurácia | **99,6%+** |
| Localização — teste cego oficial (condição ideal, 15-450 km) | MAE / mediana / P95 | 0,665 km / 0,358 km / 2,834 km |
| Localização — validação em escala (280 casos, toda a faixa de parâmetros) | Cobertura / MAE | 96,1% conclusivo / 2,61 km |
| Localização — após checagem de sanidade por ML | Falso-conclusivo | 1,49% → **0,38%** dos casos conclusivos |
| Confiança média das árvores (após calibração Platt/sigmoid) | — | ~88,7% |

Todos os números de validação independente vêm de lotes gerados **depois**
do treino, nunca reaproveitados como dado de treino (auditoria automática de
vazamento em `src/manifest.py`). Detalhes completos, incluindo casos que
falharam, estão em `resultados_experimentos/` e no histórico de versões
abaixo.

## Desempenho por classe e faixa de impedância (em andamento)

> **Nota**: esta tabela reflete um lote de testes de um modelo **experimental**
> (candidato v20, ainda em ajuste), não o modelo em produção. Serve para
> documentar o progresso do ajuste fino em direção à meta de 90%+ de acerto
> em todas as classes e faixas de resistência de falta (Rfault). Teste: 10.000
> casos gerados de forma independente (100 por classe × Rfault), nunca usados
> no treino.

| Classe | Muito baixa<br>(≤15 Ω) | Baixa<br>(50-100 Ω) | Média<br>(300 Ω) | Alta<br>(900-1800 Ω) | Muito alta<br>(2800-3000 Ω) |
|---|:---:|:---:|:---:|:---:|:---:|
| `AG`  | 100% | 100% | 100% | 100% | 100% |
| `BG`  | 100% | 100% | 100% | 100% | 100% |
| `CG`  | 100% | 100% | 100% | 100% | 100% |
| `AB`  | 🔴 61% | 100% | 100% | 100% | 100% |
| `BC`  | 🔴 76% | 100% | 100% | 100% | 99% |
| `CA`  | 🟡 93% | 100% | 100% | 100% | 100% |
| `ABG` | 100% | 97% | 95% | 🟡 84% | 🔴 72% |
| `BCG` | 100% | 100% | 98% | 🟡 89% | 🔴 64% |
| `CAG` | 100% | 100% | 100% | 98% | 🟡 90% |
| `ABC` | 100% | 100% | 100% | 100% | 🟡 88% |

🔴 abaixo de 80%&nbsp;&nbsp;🟡 80-89%&nbsp;&nbsp;(sem marcação = 90%+)

**Padrão identificado**: os dois pontos fracos têm causas físicas distintas,
não uma única "zona de fronteira":

- **`AB`/`BC` em Rfault muito baixo (≤15 Ω)**: a feature mais discriminativa
  (razão de sequência-zero da corrente) fica instável nesse regime — o
  desvio-padrão chega a ser da ordem da própria média — mesmo em faltas
  puramente fase-fase, sem ligação à terra. Não é confusão física real com
  `ABG`/`BCG`; é ruído numérico na janela de transitório. Mais dados de
  treino no mesmo formato não resolvem — é necessária uma feature mais
  robusta nesse regime (ex: janela de integração maior).
- **`ABG`/`BCG` em Rfault alto (≥900 Ω)**: degradação gradual e fisicamente
  esperada — com resistência de falta muito alta, a corrente pelo caminho de
  terra tende a zero, aproximando o sinal do de uma falta `AB`/`BC` sem
  aterramento. Mais dados de treino ajudam aqui, mas com retorno
  decrescente; pode haver um limite físico de separabilidade nos extremos
  (2800-3000 Ω).

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
├── train_distance_sanity_regressor.py  Treino da checagem de sanidade (ML)
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

docs/                       Relatório parcial entregue anteriormente
resultados_experimentos/    Saídas de experimentos (curva de aprendizado, comparação de modelos)
legado/                      Abordagem anterior (wavelet/limiar), mantida como referência histórica
tests/                       Suíte de testes automatizados
run_tests.py                 Executa toda a suíte (python run_tests.py)
```

## Como usar (produto final)

1. [Baixe o `.zip` portátil](#download-rápido) e extraia em qualquer pasta.
2. Dê dois cliques em `app/ABRIR_CLASSIFICADOR.bat` (ou direto em
   `ClassificadorFaltasATP.exe`).
3. Clique em **Escolher PL4…** e selecione o arquivo gerado pelo ATPDraw.
4. Clique em **Analisar**.
5. Veja o tipo de falta, a distância, a confiança e a forma de onda no
   gráfico.
6. Use **Salvar resultado (JSON)…** para guardar o resultado.

> O Windows SmartScreen ou seu antivírus pode alertar na primeira execução
> por ser um executável sem assinatura digital (comum em ferramentas
> acadêmicas/independentes). Clique em "Mais informações → Executar assim
> mesmo". O código-fonte completo está neste repositório para quem quiser
> auditar antes de rodar.

## Como usar (código-fonte)

Requer Python 3.11+, [Git LFS](https://git-lfs.com/) (para clonar os
arquivos grandes) e as dependências do projeto (`scikit-learn`, `numpy`,
`scipy`, `joblib`, `matplotlib`).

```powershell
git clone https://github.com/andrelsrv/IC-UFPI.git
cd IC-UFPI
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
com verificação cruzada de consistência entre os dois terminais. Um
segundo modelo de *machine learning* (`RandomForestRegressor`), treinado
para estimar a distância a partir da atenuação do sinal — um princípio
físico independente da correlação de reflexão — atua como checagem de
sanidade cruzada: se as duas estimativas discordarem muito, o resultado
vira inconclusivo em vez de arriscar uma resposta errada. Abaixo de um SNR
estimado de 50 dB, ou fora da janela temporal validada, a localização
também é bloqueada. A classificação permanece disponível de qualquer
forma, pois é o componente validado com maior robustez.

## Faixas de parâmetros validadas

| Parâmetro | Faixa | Justificativa |
|---|---|---|
| Classificação (10 classes) | 15 – 600 km | limite físico usual de linha CA sem compensação série |
| Localização | 15 – 600 km | teste cego oficial cobriu 15-450 km; 450-600 km revalidado informalmente |
| Resistência de falta (Rfault) | 0,01 – 3000 Ω | de curto franco até falta de alta impedância (vegetação/solo seco) |
| Ângulo de incidência | 0° – 360° | a falta pode ocorrer em qualquer ponto do ciclo de 60 Hz |
| Instante de fechamento (t_cl) | 0,025 – 0,675 s | livre dentro da simulação (Tmax=0,7s no template ATP) |

> **Para uso manual no ATPDraw**: se você gerar o caso manualmente pelo
> ATPDraw (fora do pipeline Python), o `Tmax` do seu projeto também precisa
> ser ajustado para 0,7s para que t_cl fora de ~100ms funcione
> corretamente. Não aumente o Tmax além disso sem revalidar: o solver ATP
> corrompe resultados acima de ~800-900 mil passos de tempo (detalhes em
> `modelos/FINAL_PIPELINE_FREEZE_V12.json`).

`ABC-G` (trifásica-terra) não é uma classe suportada: testes de viabilidade
mostraram que ela não é separável de `ABC` com confiabilidade estatística,
o que é consistente com a física do problema (uma falta trifásica simétrica
produz corrente de neutro próxima de zero, com ou sem aterramento).

## Limitações conhecidas

- **Risco residual do localizador (reduzido, não eliminado)**: mesmo com a
  checagem de sanidade por ML, ~0,4% dos casos conclusivos ainda podem
  receber uma distância errada com confiança falsa (a taxa antes dessa
  proteção era ~1,5%). Não há um filtro conhecido que elimine esse risco
  por completo — trate a distância reportada como uma estimativa, não
  como garantia absoluta.
- Há uma confusão residual muito pontual entre `ABG` e `AB` em distâncias
  muito curtas (1 caso em 300 testados originalmente); não reproduzida em
  testes posteriores — tratada como ruído estatístico, não um padrão
  sistemático corrigível.
- As validações incrementais são lotes informais pós-treino; apenas o
  teste cego oficial (v1) segue metodologia de campanha cega completa com
  splits bloqueados antes do treino.
- A faixa 600-620 km do localizador é apenas folga técnica no teto de
  busca e não foi validada com casos reais.

## Privacidade e segurança

- O aplicativo roda **inteiramente offline** — não envia dados pela
  internet, não coleta telemetria, não faz nenhuma requisição de rede.
- Todo o processamento (leitura do `.pl4`, classificação, localização)
  acontece localmente na sua máquina.
- O repositório não contém credenciais, chaves de API, tokens, nem dados
  pessoais — apenas código-fonte, modelos treinados e resultados de
  experimentos com dados simulados.
- O código-fonte completo está disponível para auditoria; nada no `.exe`
  faz algo que o código em `src/` não mostre explicitamente.

## Testes automatizados

```powershell
python run_tests.py
```

35 testes cobrindo extração de atributos, leitura de arquivos, geração de
casos, validação de manifesto e localização por ondas viajantes.

## Histórico de versões do modelo

O modelo passou por 12 iterações (v1 a v12), evoluindo de uma linha de
base com 500 casos (macro-F1 100% no teste cego oficial) até o estado
atual: mais dados de treino em faixas antes fracas (curta distância,
alta impedância, 450-600 km), calibração de confiança, correção de um
bug de normalização no localizador, uma checagem de sanidade por ML para
reduzir respostas falso-confiantes, e ampliação da janela de t_cl aceita
de 30ms para 650ms. O changelog completo, com números de cada etapa e o
motivo de cada mudança, está em `modelos/FINAL_PIPELINE_FREEZE_V12.json`.

---

<div align="center">

Desenvolvimento de pesquisa acadêmica (PIBIC) — Universidade Federal do Piauí.

</div>
