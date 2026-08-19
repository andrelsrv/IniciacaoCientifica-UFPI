<div align="center">

# ⚡ Classificador e Localizador de Faltas em Linhas de Transmissão

**Detecção, classificação e localização de faltas elétricas** a partir de
simulações eletromagnéticas transitórias (ATP/ATPDraw), usando extração de
atributos físicos e *machine learning*.

Projeto de Iniciação Científica (PIBIC) — Universidade Federal do Piauí

[![Download](https://img.shields.io/badge/⬇%20Download-Vers%C3%A3o%20portátil%20(.zip)-2ea44f?style=for-the-badge)](https://github.com/andrelsrv/IniciacaoCientifica-UFPI/releases/latest/download/ClassificadorFaltasATP-v1.0-portable.zip)
[![Releases](https://img.shields.io/github/v/release/andrelsrv/IniciacaoCientifica-UFPI?style=for-the-badge&label=vers%C3%A3o)](https://github.com/andrelsrv/IniciacaoCientifica-UFPI/releases)
[![Testes](https://img.shields.io/badge/testes-36%2F37%20passando-brightgreen?style=for-the-badge)](#testes-automatizados)
[![Licença](https://img.shields.io/badge/licença-MIT-blue?style=for-the-badge)](LICENSE)

</div>

---

## Sumário

- [Visão geral](#visão-geral)
- [Download rápido](#download-rápido)
- [Resultados](#resultados)
- [Desempenho por classe](#desempenho-por-classe)
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
   `BC`, `CA`, `ABG`, `BCG`, `CAG`, `ABC`) usando um `RandomForestClassifier`
   em duas etapas (especialista + geral) sobre 67 atributos físicos — RMS,
   picos, componentes simétricas clássicas e a **transformação modal real da
   linha** (extraída da geometria física dos condutores, não uma aproximação
   simétrica), mais um segundo nível de classificadores binários
   especialistas que arbitram os pares mais difíceis de confundir
   (`AB`/`ABG`, `BC`/`BCG`, `CA`/`CAG`).
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

**[⬇ Baixar a versão portátil (.zip)](https://github.com/andrelsrv/IniciacaoCientifica-UFPI/releases/latest/download/ClassificadorFaltasATP-v1.0-portable.zip)**

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
> não funcionaria. O `.zip` da seção [Releases](https://github.com/andrelsrv/IniciacaoCientifica-UFPI/releases)
> já vem com tudo resolvido e pronto para uso.

Se você é desenvolvedor e quer o código-fonte com histórico completo, use
`git clone` (requer [Git LFS](https://git-lfs.com/) instalado) — veja
[Como usar (código-fonte)](#como-usar-código-fonte).

## Resultados

Validação ponta-a-ponta da versão atual (**G2.2**): 25.000 simulações ATP
geradas do zero (2.500 por classe, `.pl4` reais, nunca usados no treino) e
processadas pelo mesmo executável/script que o usuário final roda —
não um atalho estatístico que chama só o classificador.

| Etapa | Métrica | Valor |
|---|---|---|
| Classificação (25.000 casos reais, 10 classes) | Acurácia geral | **97,1%** |
| Localização (mesmo lote) | Conclusivo / MAE / mediana | 82,1% / 2,19 km / 0,08 km |
| Classificação — teste cego oficial histórico (70 casos, 7 condições) | Macro-F1 | 100% |

Todos os números de validação vêm de lotes gerados **depois** do treino,
nunca reaproveitados como dado de treino (auditoria automática de vazamento
em `src/manifest.py`). Detalhes completos, incluindo o histórico de todas as
gerações do modelo, estão em
[`modelos/FINAL_PIPELINE_FREEZE_G2.2.json`](modelos/FINAL_PIPELINE_FREEZE_G2.2.json).

## Desempenho por classe

> Teste de 25.000 casos reais (2.500 por classe), modelo em produção (**G2.2**).

| Classe | Acurácia |
|---|:---:|
| `AG` | 100,0% |
| `BG` | 100,0% |
| `CG` | 100,0% |
| `ABG` | 99,7% |
| `ABC` | 99,5% |
| `BCG` | 99,3% |
| `CAG` | 99,2% |
| `CA` | 94,9% |
| `BC` | 93,4% |
| `AB` | 🟡 85,3% |

🟡 abaixo de 90% (sem marcação = 90%+)

> ⚠️ **Sobre `Rfault` em faltas fase-fase**: `Rfault` só é fisicamente
> aplicável a faltas que envolvem terra (`AG`/`BG`/`CG`/`ABG`/`BCG`/`CAG`).
> Faltas puramente fase-fase (`AB`/`BC`/`CA`/`ABC`) não têm caminho físico
> até o resistor de falta — na prática são sempre francas (~0 Ω), e o
> gerador de casos (`src/fault_case_generator.py`) rejeita qualquer valor de
> `Rfault` diferente de zero para essas 4 classes.

**Ponto fraco restante: `AB`.** É a classe mais difícil desde o início do
projeto (physicamente, a fase B fica no meio da disposição horizontal dos
condutores da linha — ver `modelos/FINAL_PIPELINE_FREEZE_G2.1.json` para a
investigação completa dessa assimetria física). Já passou por três rodadas
de melhoria (correção da topologia de `Rfault`: +13 pontos; feature de
transformação modal real: +13 pontos; reativação do classificador
especialista: +9 pontos), saindo de ~57% para 85%. Ideias futuras: feature
de onda viajante (ver `docs/`), ou aceitar como teto físico e documentar.

## Estrutura do projeto

```
app/                        Produto final para uso direto
├── ClassificadorFaltasATP.exe   Aplicativo standalone (não precisa de Python)
├── classificador_config.json    Aponta para o modelo ativo em modelos/
└── ABRIR_CLASSIFICADOR.bat      Atalho de execução

src/                         Código-fonte Python (pipeline em produção)
├── classificador_gui_v2.py       Interface gráfica (com gráfico da forma de onda)
├── infer_fault.py                Inferência via linha de comando
├── feature_extraction.py         Extração dos 67 atributos físicos
├── pl4_reader.py / signal_io.py  Leitura de arquivos .pl4 / .adf
├── traveling_wave_localizer.py   Localização por correlação de ondas viajantes
├── adaptive_localizer.py         Localização multi-banda com verificação de SNR
├── multiscale_localizer.py       Localização multi-escala
├── train_distance_sanity_regressor.py  Treino da checagem de sanidade (ML)
├── train_robust_classifier_v31.py      Treino do classificador em produção
├── calibrate_modal_final.py            Calibração de confiança em produção
├── fault_case_generator.py       Geração de casos de falta no ATP
├── jmarti_generator.py / simulation_generator.py / rebuild_reference_case.py
│                                  Geração e reconstrução de simulações ATP
├── manifest.py / cached_manifest.py / precompute_manifest_features.py
│                                  Manifesto de treino e cache de atributos, com
│                                  auditoria anti-vazamento
├── pilot_planner.py / pilot_runner.py  Planejamento e execução de campanhas
│                                        de geração com splits bloqueados
├── evaluate_blind_test.py        Executor de campanha de teste cego oficial
├── robustness_evaluation.py      Perturbações (ruído/ganho/sync) para teste de robustez
├── legado_experimentos/          Scripts de versões anteriores (histórico de pesquisa,
│                                  fora do pipeline ativo — ver README interno)
└── ABRIR_EM_MODO_DESENVOLVEDOR.bat  Roda a GUI a partir do código-fonte (requer Python)

modelos/                     Artefatos em produção
├── robust_classifier_G2.1_calibrated.joblib   Classificador ativo (com especialistas)
├── distance_sanity_regressor.joblib           Checagem de sanidade do localizador (ML)
├── FINAL_PIPELINE_FREEZE_G2.2.json            Congelamento ativo + histórico completo
├── FINAL_PIPELINE_FREEZE_G2.0.json / G2.1.json  Gerações anteriores (referenciadas em G2.2)
├── classificador_config.master.json
└── historico/                 Modelos e congelamentos anteriores à geração G2 (ver README interno)

docs/                         Relatório parcial entregue anteriormente
resultados_experimentos/      Saídas de experimentos (curva de aprendizado, comparação de modelos)
legado/                       Abordagem anterior (wavelet/limiar), mantida como referência histórica
tests/                        Suíte de testes automatizados
run_tests.py                  Executa toda a suíte (python run_tests.py)
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
git clone https://github.com/andrelsrv/IniciacaoCientifica-UFPI.git
cd IniciacaoCientifica-UFPI
cd src
python classificador_gui_v2.py
```

Ou, para inferência direta em linha de comando:

```powershell
cd src
python infer_fault.py "C:\caminho\novo.pl4" `
  --classifier "..\modelos\robust_classifier_G2.1_calibrated.joblib" `
  --freeze "..\modelos\FINAL_PIPELINE_FREEZE_G2.2.json" `
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
A partir desse instante são extraídos 67 atributos: razões RMS e de pico
antes/depois do evento, degrau transitório máximo, razões de componentes
simétricas clássicas (sequência zero/positiva/negativa) de tensão e
corrente em ambos os terminais, assimetria entre fases, e a razão entre
modos da **transformação modal real da linha** — a matriz de autovetores
extraída do próprio cálculo de constantes de linha do ATP para a geometria
física dos condutores (disposição horizontal, fase B no meio, não uma torre
simétrica), que captura uma assimetria elétrica real que a aproximação
clássica de componentes simétricas não vê.

### Classificação

Um `RandomForestClassifier` em duas etapas (via `warm_start`: árvores
especialistas em dados de alta impedância + árvores gerais) é treinado
sobre dezenas de milhares de casos simulados em 10 classes de falta,
selecionado pelo pior caso de F1-macro em 7 condições de robustez (ruído,
erro de ganho, erro de sincronização) — comparado também contra
ExtraTrees, Gradient Boosting e redes neurais MLP. A confiança das árvores
é recalibrada com `CalibratedClassifierCV` (método sigmoid/Platt). Um
segundo nível de classificadores binários especialistas (`AB` vs `ABG`,
`BC` vs `BCG`, `CA` vs `CAG`) arbitra a decisão final sempre que o
classificador geral prevê uma dessas classes — esses pares são os mais
fáceis de confundir entre si, e um classificador dedicado só a eles
consistentemente acerta mais que o classificador geral de 10 classes.

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
| Resistência de falta (Rfault) | 0,01 – 3000 Ω para faltas com terra (`AG`/`BG`/`CG`/`ABG`/`BCG`/`CAG`); faltas puramente fase-fase (`AB`/`BC`/`CA`/`ABC`) são sempre francas (~0 Ω) | Rfault só tem caminho físico até o terra; faltas fase-fase não têm esse caminho |
| Ângulo de incidência | 0° – 360° | a falta pode ocorrer em qualquer ponto do ciclo de 60 Hz |
| Instante de fechamento (t_cl) | 0,0833 – 0,09995 s (~83,3-100ms) para o que foi efetivamente gerado no treino | corresponde a 5 ciclos de 60Hz + um ângulo de incidência de 0-360° dentro do 6º ciclo — cobre todos os ângulos possíveis, mas só nesse instante absoluto |

> **t_cl fora dessa janela**: testes pontuais mostraram que o classificador
> generaliza bem para t_cl fora da faixa treinada (testado até 450ms) **na
> maioria das distâncias** — mas não em todas. Ver a nota sobre
> "covas" distância×ângulo em [Limitações conhecidas](#limitações-conhecidas):
> em certas combinações estreitas e específicas de distância, mudar o t_cl
> pode derrubar a acurácia de `AB`/`BC` mesmo dentro da janela treinada.
> Não assuma que qualquer combinação de distância dentro de 15-600km vai
> funcionar bem só porque a média geral do projeto é alta — cada distância
> nova usada num caso real (fora dos lotes de teste já rodados) deveria, a
> rigor, ser validada especificamente antes de confiar no resultado.

> **Para uso manual no ATPDraw**: se você gerar o caso manualmente pelo
> ATPDraw (fora do pipeline Python), o `Tmax` do seu projeto também precisa
> ser ajustado para 0,7s para que t_cl fora de ~100ms funcione
> corretamente. Não aumente o Tmax além disso sem revalidar: o solver ATP
> corrompe resultados acima de ~800-900 mil passos de tempo (detalhes em
> `modelos/historico/FINAL_PIPELINE_FREEZE_V12.json`).

`ABC-G` (trifásica-terra) não é uma classe suportada: testes de viabilidade
mostraram que ela não é separável de `ABC` com confiabilidade estatística,
o que é consistente com a física do problema (uma falta trifásica simétrica
produz corrente de neutro próxima de zero, com ou sem aterramento).

## Limitações conhecidas

- **`AB` continua sendo a classe mais fraca** (85,3%), apesar de três
  rodadas de melhoria consecutivas. Ver [Desempenho por classe](#desempenho-por-classe).
- **"Covas" de distância×ângulo em `AB`/`BC`**: descoberto ao comparar
  contra um projeto de referência externo (base enviada pelo orientador,
  linha de 200km com falta a 90km do terminal local) que, nessa distância
  exata, `BC` caía para ~39-55% de acerto (bem abaixo da média de ~94%) —
  não por causa da distância isoladamente, mas de uma combinação estreita
  e específica de distância + ângulo de incidência. Variando só a
  distância (mantendo ângulo aleatório) a acurácia oscila de forma não
  suave entre pontos vizinhos (ex.: BC=100% a 110km de distância do PDT,
  mas 47-53% a 90km e 130km, a só 20km de diferença) — um padrão
  consistente com reflexão de onda chegando num timing que atrapalha as
  features de sequência-zero/modal, não com falta de dados de treino numa
  região ampla. Tentativas de corrigir via mais dados de treino (ângulo
  denso, sistemático) consertam os pontos exatos treinados (ex.: a
  distância do orientador foi de 39-78% para 100% em 5.000 casos de
  validação) mas **não generalizam** para distâncias vizinhas não
  treinadas (testada uma vizinha a 20km de distância, sem melhora). Ou
  seja: **não assuma que qualquer distância dentro de 15-600km tem alta
  acurácia** — só as distâncias efetivamente testadas têm essa garantia.
  Resolver isso de forma geral provavelmente exige uma feature nova,
  menos sensível a esse tipo de coincidência de reflexão (candidata:
  feature explícita de onda viajante), não mais dados de treino pontuais.
- **Localização em faltas muito distantes (>450 km)**: cerca de 3% desses
  casos recebem uma distância com erro grande (dezenas a centenas de km)
  por provável falsa reflexão — a classificação nesses casos continua
  correta, só a distância calculada erra. Investigação aprofundada ainda
  não feita.
- **Risco residual do localizador em geral**: mesmo com a checagem de
  sanidade por ML, uma pequena fração dos casos conclusivos ainda pode
  receber uma distância errada com confiança falsa. Trate a distância
  reportada como uma estimativa, não como garantia absoluta.
- O regressor de checagem de sanidade de distância não foi retreinado
  desde a primeira geração (G1) — não conhece as features mais recentes
  (excluídas explicitamente antes de chamá-lo, ver `infer_fault.py`).
- As validações incrementais são lotes informais pós-treino; apenas o
  teste cego oficial (histórico, v1) segue metodologia de campanha cega
  completa com splits bloqueados antes do treino. Recomenda-se rodar uma
  nova campanha cega oficial com a versão G2.2 antes de qualquer alegação
  de rigor cego total.
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

37 testes cobrindo extração de atributos, leitura de arquivos, geração de
casos, validação de manifesto e localização por ondas viajantes. 36
passam; 1 falha conhecida e pré-existente
(`test_abg_uses_two_independent_resistors_to_ground`), não relacionada às
mudanças recentes — documentada, não corrigida por estar fora do escopo
das últimas rodadas de trabalho.

## Histórico de versões do modelo

O projeto passou por duas fases de nomenclatura. Até a versão 29 (v1-v29),
numeração sequencial simples. A partir daí, reorganizado em **gerações**
(G1, G2.0, G2.1, G2.2...) reservadas para promoções reais a produção —
experimentos de pesquisa passaram a usar nomes descritivos em vez de
números (ver `src/legado_experimentos/`).

Marcos principais:

- **G1** (histórico, ~v1-v29): evolução de uma linha de base com 500 casos
  até um pipeline com 10 classes, calibração de confiança, correção de um
  bug de normalização no localizador, checagem de sanidade por ML, e
  ampliação da janela de t_cl aceita.
- **G2.0**: corrigida a topologia de `Rfault` em faltas fase-fase — o
  gerador de casos inseria um resistor onde a física real da linha não tem
  esse caminho (bolted switch, sem resistor). Bug que afetava toda a base
  de dados histórica de `AB`/`BC`/`CA`/`ABC`.
- **G2.1**: descoberta de que os condutores da linha estão numa disposição
  horizontal assimétrica (fase B no meio), o que explica por que `AB`/`BC`
  eram sistematicamente mais fracas que `CA` em toda versão anterior.
  Adicionada uma feature baseada na transformação modal real da linha
  (extraída do cálculo de constantes do ATP): `AB` 56,9%→70,2%, `BC`
  71,4%→84,4%.
- **G2.2** (atual): reavaliados os classificadores especialistas binários
  `AB`/`ABG` e `BC`/`BCG` (desligados desde a G2.0 por piorarem o
  resultado) à luz da feature modal nova — dessa vez ajudaram de verdade:
  `AB` 70,2%→85,3%, `BC` 84,4%→93,4% (números do teste real de 25k casos).

O changelog completo de cada geração, com números detalhados e o motivo de
cada mudança, está em `modelos/FINAL_PIPELINE_FREEZE_G2.2.json` (que
referencia G2.1, que referencia G2.0, e assim por diante) e em
`modelos/historico/` para as versões pré-G2.

---

<div align="center">

Desenvolvimento de pesquisa acadêmica (PIBIC) — Universidade Federal do Piauí.

</div>
