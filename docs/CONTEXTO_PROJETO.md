# Contexto do projeto — localização e classificação de faltas no ATPDraw

## Finalidade deste documento

Este arquivo registra o contexto e as decisões da conversa de projeto para que o trabalho possa ser retomado em outro computador ou em uma nova sessão do Codex.

Ao retomar, leia este documento e também `ClassificadorATP.py`. Não implemente mudanças antes de terminar a definição dos requisitos pendentes com o usuário.

## Objetivo principal

Desenvolver um código capaz de analisar sinais exportados de simulações do ATPDraw e retornar, com a maior assertividade possível:

1. o tipo de falta elétrica;
2. a localização da falta em quilômetros, medida a partir do terminal PDT.

Exemplo de saída desejada:

```text
Falta AG localizada a 229,7 km de PDT
```

O circuito representa uma linha de transmissão entre os terminais PDT e BEA. O ponto de falta está na junção entre duas seções LCC. Na imagem discutida, as seções possuem 230 km e 70 km, colocando a falta a 230 km de PDT.

## Variáveis das simulações

A campanha de simulações deverá variar pelo menos:

- distância da falta a partir de PDT, alterando os comprimentos das seções LCC;
- resistência de falta (`Rfault`);
- instante de fechamento das chaves (`t_close`), incluindo diferentes ângulos de incidência da falta;
- combinação das chaves, produzindo diferentes tipos de falta;
- comprimento do trecho posterior à falta, tratado como condição variável do sistema e não como saída de interesse.

O comprimento total da linha não é uma saída relevante para o usuário. O objetivo é estimar somente a distância entre PDT e o ponto da falta.

## Decisões já confirmadas

### Referência da localização

- PDT será a origem, correspondente ao km 0.
- O resultado será expresso em quilômetros a partir de PDT.
- Mesmo que sinais dos dois terminais sejam usados, haverá uma única coordenada final referenciada a PDT.

### Faixa inicial de validação

- A primeira versão será validada para faltas entre 1 km e 500 km de PDT.
- O algoritmo físico poderá produzir estimativas fora dessa faixa, mas deverá identificá-las como fora do domínio validado.
- A faixa poderá ser ampliada depois de medir o erro da primeira campanha.

### Passo de simulação

- Usar passo temporal fixo de `1 µs` em todas as simulações de treinamento, validação e teste.
- Para velocidade de propagação próxima de `295.000 km/s`, a resolução temporal teórica do cálculo de ida e volta é aproximadamente 0,148 km por amostra:

```text
delta_d = velocidade * delta_t / 2
```

- Essa resolução teórica não é uma garantia do erro real, que também depende de dispersão, ruído, modelo da linha e detecção correta das frentes de onda.

### Medições

O usuário aceitou adicionar medições nos dois terminais para maximizar a assertividade.

A configuração recomendada é exportar, todos na mesma base de tempo do ATPDraw:

- `VA`, `VB`, `VC`, `IA`, `IB`, `IC` no terminal PDT;
- `VA`, `VB`, `VC`, `IA`, `IB`, `IC` no terminal BEA.

Os nomes definitivos e a ordem das colunas ainda precisam ser documentados depois que os medidores forem configurados no circuito.

### Proibição de vazamento de rótulos

A classificação não pode usar o nome do arquivo, caminho, nome da pasta ou qualquer convenção de nomenclatura para deduzir o tipo ou a distância da falta.

Na inferência, as entradas permitidas serão somente os sinais elétricos e parâmetros físicos legítimos que forem aprovados posteriormente.

Os rótulos verdadeiros deverão ficar em um manifesto separado, utilizado exclusivamente para:

- montar o conjunto supervisionado;
- dividir treino, validação e teste;
- calcular métricas.

Antes de entregar dados ao modelo, devem ser removidos:

- nomes e caminhos de arquivos;
- índices de execução que codifiquem a classe;
- distância verdadeira;
- `Rfault` verdadeira;
- `t_close` verdadeiro;
- estados de chave que revelem diretamente o rótulo.

O classificador deve reconhecer a falta pela análise dos sinais, não por metadados que contenham a resposta.

## ATPDraw e automação

O usuário acredita estar utilizando ATPDraw 7.7.

Nessa versão, a campanha pode ser automatizada com o Internal Parser, variáveis em função de `KNT`, laços e/ou tabelas externas por `@FILE`.

Devem ser parametrizados:

- comprimento das seções `_LCC`;
- `Rfault`;
- `t_close` e, se necessário, `t_open`;
- estados/tempos das chaves responsáveis pelo tipo de falta.

Uma tabela de geração poderá ter estrutura semelhante a:

```text
distancia_km  trecho_remoto_km  rfault_ohm  tclose_s  sw_ag  sw_bg  sw_cg  sw_ab  sw_bc  sw_ca
50            250               0.01         0.0800    1      0      0      0      0      0
125           175               10.0         0.0835    0      0      0      1      0      0
230           70                50.0         0.0870    1      1      1      0      0      0
```

Essa tabela é apenas ilustrativa. A codificação real das chaves deve ser confirmada a partir do circuito e não deve ser fornecida ao classificador durante a inferência.

Referência consultada: [documentação oficial de variáveis do ATPDraw](https://www.atpdraw.net/help7/html_variables.html).

## Estratégia técnica recomendada

Usar uma arquitetura híbrida, em vez de depender somente de limites manuais ou somente de aprendizado de máquina.

### Detecção

- detectar automaticamente o início da perturbação;
- separar falta de operação normal e outros transitórios;
- criar janelas pré-falta e pós-falta coerentes.

### Localização

- extrair os tempos de chegada das ondas viajantes;
- combinar evidências dos terminais PDT e BEA;
- usar tensões e correntes, transformações modais e detecção robusta das primeiras frentes;
- não selecionar simplesmente os dois primeiros picos;
- calibrar ou estimar a velocidade efetiva de propagação do modelo LCC;
- produzir estimativa em km e indicador de incerteza;
- avaliar erro absoluto médio, mediano, percentis e pior caso por distância, classe e `Rfault`.

Como o comprimento total não será necessariamente fornecido, a formulação final da localização precisa ser estudada com cuidado. Pode ser necessário usar localização de terminal único em cada extremidade e fusão das estimativas, ou estimar informações adicionais pelas reflexões. Não assumir automaticamente a fórmula de dois terminais que exige comprimento total conhecido.

### Classificação

- usar sinais trifásicos de tensão e corrente;
- extrair componentes de sequência e/ou componentes modais em janelas pré e pós-falta;
- incluir atributos normalizados para reduzir dependência de amplitude e `Rfault`;
- comparar métodos interpretáveis e modelos supervisionados;
- calibrar probabilidades e permitir resultado inconclusivo quando a confiança for insuficiente;
- apresentar matriz de confusão e métricas por classe em dados cegos.

### Separação dos dados

Não fazer uma divisão aleatória ingênua entre formas de onda quase idênticas.

O conjunto de teste deve conter combinações não vistas de:

- distância;
- `Rfault`;
- `t_close`/ângulo de incidência;
- tipo de falta;
- condições do trecho remoto.

Preferir divisão por grupos ou blocos de parâmetros, evitando que variações mínimas do mesmo cenário apareçam simultaneamente em treino e teste.

## Problemas identificados no código atual

Arquivo analisado: `ClassificadorATP.py`.

### Leitura incompleta do ADF

O carregador aceita somente as quatro primeiras colunas numéricas e as nomeia como `t`, `VA`, `VB`, `VC`. A captura mostrada pelo usuário possui 16 variáveis. Assim, correntes e outros terminais são descartados.

### Localização frágil

O código:

1. calcula energia wavelet;
2. escolhe o primeiro pico como `t1`;
3. escolhe o próximo pico com separação mínima de 45 µs como `t2`;
4. usa `d = velocidade * (t2 - t1) / 2`.

Não existe evidência de que `t2` seja realmente a reflexão correta. Ele pode representar chaveamento, ruído, outra reflexão ou resposta modal. O limite fixo de 45 µs também exclui distâncias curtas e não é fundamentado no circuito.

### Classificação por limites arbitrários

A confiança retornada (`0.95`, `0.88`, etc.) é fixa e não foi calibrada com dados. Portanto, não representa probabilidade ou assertividade medida.

As regras de identificação de fases e faltas bifásicas precisam ser revisadas, pois algumas comparações entre RMS e rótulos retornados são inconsistentes.

### Componentes simétricas

A função recebe séries temporais inteiras. A multiplicação matricial produz séries complexas, o que pode funcionar dimensionalmente, mas a avaliação RMS deve ser confirmada e feita nas janelas corretas, preferencialmente a partir de fasores ou de uma formulação temporal claramente documentada.

### FFT e ângulos incorretos

O código aplica `np.abs(fft(...))` antes de calcular os ângulos. Depois do módulo, a parte imaginária foi perdida; por isso, os ângulos calculados não representam as fases elétricas.

Também usa o índice da fundamental encontrado somente na fase A para todas as fases e pode selecionar o componente DC como fundamental.

### THD simplificada incorreta

O cálculo soma bins posteriores ao maior bin sem separar frequências positivas/negativas, vazamento espectral, fundamental e harmônicos. Esse resultado não deve ser chamado de THD sem correção.

### Janela e índice zero

O trecho `if idx_t1` trata o índice zero como falso. Se o evento for detectado na primeira amostra, a janela será definida incorretamente.

### Código e dependências

- Há imports não utilizados, como `RandomForestClassifier`, `StandardScaler` e `joblib`.
- Não há pipeline de treinamento ou modelo persistido, apesar desses imports.
- Existe texto inválido ao final do arquivo: `/* ULTIMO TESTE DE GIT COMMIT V1 */`, que não é comentário válido em Python.
- O texto aparece com problemas de codificação de caracteres, indicando possível mistura entre UTF-8 e outra codificação.

## Classes discutidas — histórico superado pela decisão final

Foi proposta a seguinte taxonomia inicial de 12 classes:

```text
Sem falta
AG, BG, CG
AB, BC, CA
ABG, BCG, CAG
ABC
ABCG
```

A classe `Sem falta` é recomendada para que chaveamentos e perturbações normais não sejam obrigatoriamente classificados como falta.

`ABC` e `ABCG` serão inicialmente separadas, mas a separação só deve ser mantida se os testes cegos demonstrarem que os sinais disponíveis permitem distingui-las de maneira confiável. Caso contrário, a saída tecnicamente defensável será `ABC/ABCG`.

Essa proposta foi posteriormente testada e substituída pela decisão final registrada abaixo.

## Pergunta histórica já resolvida

Pergunta que orientou a validação:

> Você aprova o conjunto inicial de 12 classes: Sem falta; AG, BG, CG; AB, BC, CA; ABG, BCG, CAG; ABC; ABCG?

O pré-teste confirmou que `ABC` e `ABCG` não são separáveis de maneira fisicamente defensável no sistema equilibrado; elas foram consolidadas em `ABC`.

## Questões posteriores a resolver, uma por vez

Após fechar as classes, ainda será necessário decidir:

1. faixa e distribuição de `Rfault`;
2. faixa e estratégia de variação de `t_close`/ângulo de incidência;
3. passo de variação das distâncias e seleção de distâncias cegas;
4. comprimento mínimo e distribuição do trecho remoto;
5. modelo LCC usado e parâmetros dependentes da frequência;
6. presença de ruído, erro de medição e variações de fonte/carga;
7. duração pré-falta e pós-falta das simulações;
8. formato de saída (`PL4`, `ADF` ou outro) e cabeçalho real das colunas;
9. metas quantitativas de desempenho para classificação e localização;
10. método de automação e geração do manifesto de rótulos;
11. protocolo de teste cego e prevenção de vazamento de dados;
12. formato final da interface do programa.

## Regra de trabalho acordada

A conversa vinha sendo conduzida como uma sessão de revisão rigorosa de requisitos: resolver uma decisão por vez, sempre apresentando uma recomendação, e não alterar o código até que usuário e assistente confirmem entendimento compartilhado suficiente para implementar.

## Requisitos consolidados em 1 de agosto de 2026

Usuário e assistente confirmaram entendimento compartilhado suficiente para iniciar a implementação.

### Escopo

- Toda entrada terá necessariamente uma falta; `Sem falta` foi removida.
- Dez classes finais: `AG`, `BG`, `CG`, `AB`, `BC`, `CA`, `ABG`, `BCG`, `CAG` e `ABC`.
- `ABC` representa a falta trifásica. A variante `ABCG` foi consolidada nela após o pré-teste mostrar sinais praticamente idênticos, como esperado em um sistema trifásico equilibrado.
- `Inconclusiva` será usada para baixa confiança calibrada, discordância entre terminais, incerteza excessiva ou sinais inválidos/fora do domínio. Os limiares virão da validação.

### Simulações

- Frequência fixa: `60 Hz`; passo: `1 µs`; duração: `150 ms`.
- Falta-base em `83,333 ms`, apó cinco ciclos.
- Doze ângulos em passos de 30 graus (deslocamentos de aproximadamente `1,3889 ms`), deixando ao menos cerca de três ciclos pós-falta.
- `Rfault`: `0,01` a `100 Ω`, em escala aproximadamente logarítmica. Referências: `0,01; 0,1; 0,5; 1; 5; 10; 25; 50; 75; 100 Ω`; valores intermediários serão reservados.
- Linha JMarti dependente da frequência nas duas seções, com os mesmos parâmetros físicos e variação somente dos comprimentos.

### Distâncias

- Domínio validado: 1 a 500 km desde PDT.
- Treino/calibração em grade principal de 10 km, acrescida de 1 e 5 km; teste com valores intermediários.
- 1 e 5 km são distâncias da falta, não comprimentos totais de linha.
- Comprimento total mínimo: 100 km.
- Trecho posterior até 500 km. Referências de treino: `50`, `150`, `300`, `500 km`, ajustadas pelo comprimento total mínimo; valores como `100`, `225`, `400 km` poderão ser reservados.

### Entradas

```text
t_s
PDT_VA_V, PDT_VB_V, PDT_VC_V
PDT_IA_A, PDT_IB_A, PDT_IC_A
BEA_VA_V, BEA_VB_V, BEA_VC_V
BEA_IA_A, BEA_IB_A, BEA_IC_A
```

- Tempo em segundos, tensão em volts, corrente em ampères e fases na ordem A-B-C.
- Correntes positivas entrando na linha nos dois terminais; inverter BEA no carregamento se o medidor tiver orientação oposta.
- `ABC2PHR/CAR2POL` pode permanecer como ferramenta auxiliar, mas `re1-re3` e outros fasores não serão entradas. Atributos serão calculados em Python a partir dos sinais brutos.

### PL4

- PL4 será o formato principal; ADF servirá para inspeção e validação cruzada amostra por amostra.
- O leitor deve identificar os 12 canais, preservar tempo, passo, unidades e sinais, e rejeitar arquivos vazios, truncados ou com quantidade inesperada de canais.
- O arquivo de prova `C:\RESULTPESQUISA\SIMULACAOUSADA.pl4` ainda usa `5 µs`, `Tmax=0,5 s` e contém seis tensões, três correntes de PDT e `re1-re3`.
- Antes da referência final, adicionar corrente trifásica em BEA e retirar `re1-re3` das saídas.

### Robustez

- Primeiro sinais ideais; depois, por amostragem: SNR `60/40/30 dB`, ganho `±1%`, sincronização entre terminais `±1 µs`, magnitude das fontes `±5%` e impedâncias equivalentes `±10%`.
- Frequência continuará fixa. Harmônicos, saturação de instrumentos e descargas atmosféricas ficam para extensões posteriores.

### Campanha e teste

- Sem produto cartesiano completo. Piloto de cerca de 50 casos por classe.
- Referência inicial por classe: 600 treino, 150 validação e 150 teste; aproximadamente 9.900 casos no total.
- Cobertura estratificada; ampliar apenas se curvas de aprendizado justificarem.
- Internal Parser, `KNT` e tabela de parâmetros em lotes de 100 a 500.
- Arquivos brutos podem ter nomes descritivos, mas o pipeline usará `run_id` opaco. Manifesto separado guardará caminho, classe, distância, `Rfault`, ângulo, trecho remoto, robustez e divisão.
- Nomes, caminhos e rótulos não chegarão ao modelo; testes automatizados verificarão isso.
- Haverá teste de combinações inéditas e teste forte com valores físicos ausentes do treino. Ajustes usarão somente treino/validação; qualquer ajuste apó o teste exige novo conjunto cego.

### Metas

- Localização: MAE `<= 1 km`, mediana `<= 0,5 km`, P95 `<= 3 km`, pior caso `<= 5 km`.
- Classificação: macro-F1 `>= 98%` ideal, `>= 95%` robustez e recall `>= 95%` por classe.
- São objetivos de engenharia, revisáveis de forma documentada apó o piloto.

### Interface

- Biblioteca Python independente da GUI; CLI unitária e em lote; saídas texto, JSON e CSV; GUI simples opcional.

### Padronização JMarti — 2 de agosto de 2026

- A execução textual do caso atual com os `.pch` originais reproduziu o PL4 exatamente, amostra por amostra.
- O `.acp` armazena coeficientes JMarti pré-calculados; extraí-los novamente não significa recalcular o fitting.
- O usuário aprovou não misturar esses artefatos antigos com os novos.
- Todos os trechos do dataset, inclusive os comprimentos de referência, serão recalculados pelo mesmo `C:\ATP\atpmingw\tpbig.exe` autorizado.
- Cada `.pch` novo terá registro de proveniência com parâmetros e hashes SHA-256 do solver, entrada ATP e saída PCH.
- A estratégia mínima mista foi rejeitada após o primeiro pré-lote, pois o efeito da ligação à terra não apareceu adequadamente junto às chaves fase-fase.
- Configuração aterrada corrigida: `ABG=AG+BG`, `BCG=BG+CG` e `CAG=CG+AG`. Faltas aterradas usam exclusivamente as chaves fase-terra ligadas ao nó comum de `Rfault`.
- As chaves de falta ficam fechadas até depois de `Tmax` (`Top=2 s`). Com `Top` vazio, o ATP reabria cada chave na primeira passagem da corrente por zero e descaracterizava faltas multifásicas.
- O pré-teste confirmou que `ABC` e `ABCG` são praticamente indistinguíveis no sistema equilibrado. O usuário aprovou consolidá-las em `ABC`.
- Piloto pareado final: 50 combinações físicas repetidas nas 10 classes, com 35 cenários de treino, 8 de validação e 7 de teste forte; total de 500 execuções (350/80/70).
- A rede de falta textual foi corrigida para que `Rfault` participe de todas as classes: fase-terra por resistores à terra, fase-fase por resistor entre as fases e `ABC` por três resistores iguais em estrela flutuante.
- O pré-teste final de 10 classes confirmou as fases esperadas, 12 canais finitos e influência mensurável de `Rfault` também em `AB` e `ABC`.
- Campanha piloto final concluída em `pilot/campaign_v4`: 500/500 PL4 válidos, aproximadamente 3,97 GB, com 350 casos de treino, 80 de validação e 70 de teste cego; 50 casos por classe.
- A inspeção reabriu todos os PL4 e confirmou 12 canais, 150.001 amostras, passo de aproximadamente `1 µs` e valores finitos.
- O manifesto corresponde exatamente ao plano bloqueado. SHA-256: plano `60111d22abfe9fd32f0f7a68c8919974d368368f79650009c671ce8abd80f931`; manifesto `61939156e98ce96c2b69fbf0f28f901c490a2a1a7bd2f83d84e8d2cf722cf9d7`; inspeção `5602f547afa8563cacd63c245cd7e8fcb94961f0a912a98ff7593faaa162dea6`.

### Baseline de aprendizado — 2 de agosto de 2026

- `feature_extraction.py` detecta automaticamente a perturbação e calcula 61 atributos dos 12 sinais: razões RMS/pico, energia AC pós-falta, passos transitórios, componentes de sequência e atraso de chegada BEA-PDT.
- `train_baseline.py` interrompe a ingestão antes de resolver ou abrir qualquer PL4 marcado como `test_unseen`; caminhos, `run_id`, divisão, rótulo e parâmetros físicos não entram na matriz `X`.
- Foram usados 350 casos de treino e 80 de validação. Os 70 casos cegos permanecem fechados.
- Classificação ideal na validação: acurácia `100%`, macro-F1 `100%` e recall `100%` nas dez classes. Esse resultado vale apenas para o piloto ideal e ainda não comprova robustez a ruído/erros.
- Localização na validação: MAE `40,317 km`, mediana `37,6 km`, P95 `70,24 km` e máximo `144,86 km`; desempenho muito abaixo das metas.
- Conclusão: manter o classificador como baseline, mas substituir a localização genérica por uma etapa física dedicada a frentes/reflexões e estimação conjunta das distâncias, sem consultar o teste cego durante o desenvolvimento.

### Localizador por ondas viajantes — 2 de agosto de 2026

- Foi implementada separação modal/direcional das ondas incidentes e refletidas, com pareamento físico das reflexões observadas em PDT e BEA.
- A velocidade foi calibrada exclusivamente no treino em `0,2992746507 km/µs`. O comprimento total e o trecho remoto não são entradas do localizador.
- Nos 80 casos de validação, 79 foram conclusivos (`98,75%` de cobertura) e um `CG` foi corretamente rejeitado por correlação insuficiente em PDT.
- Entre os resultados conclusivos: MAE `0,543 km`, mediana `0,355 km`, P95 `2,244 km` e máximo `3,413 km`. Todas as metas iniciais de localização foram atingidas na validação ideal.
- Sem a regra de inconclusão, o caso de reflexão fraca escolheria um retorno incorreto e produziria erro de `196,6 km`; por isso ele não pode ser silenciosamente incluído como estimativa válida.
- Os 70 casos `test_unseen` continuam fechados. O próximo estágio é validar robustez a ruído, ganho e sincronização antes de qualquer avaliação cega.

### Primeira avaliação de robustez — 2 de agosto de 2026

- Foram avaliadas deterministicamente 560 combinações em memória: 80 casos de validação sob condição ideal, SNR `60/40/30 dB`, ganho independente até `±1%`, sincronização `±1 µs` e cenário combinado severo.
- Ganho `±1%`: classificação `100%`, cobertura da localização `98,75%`, MAE conclusivo `0,543 km`.
- Sincronização `±1 µs`: classificação `100%`, cobertura `97,5%`, MAE conclusivo `0,579 km`.
- SNR `60 dB`: classificação `100%`, cobertura `95%`, MAE `0,610 km`, P95 `2,463 km`; pior caso `5,083 km`.
- SNR `40 dB`: macro-F1 `85,50%`; cobertura do localizador `61,25%`. Há ainda falsa confiança em pelo menos uma reflexão, com máximo conclusivo de `55,90 km`.
- SNR `30 dB`: macro-F1 `64,52%`; cobertura `33,75%`; localização não confiável. No cenário combinado, macro-F1 `61,99%` e cobertura `32,5%`.
- Conclusão: o pipeline ideal é robusto a ganho e sincronização especificados e aceitável em `60 dB`, mas não atende às metas em `40/30 dB`. Antes do teste cego, o treinamento deve receber augmentação ruidosa e o localizador deve substituir a diferença bruta por decomposição/filtragem multiescala, além de recalibrar a regra de inconclusão.

### Pipeline final e teste cego — 3 de agosto de 2026

- O classificador foi treinado com 2.450 exemplos derivados somente dos 350 casos de treino (sete condições). Obteve macro-F1 `100%` em todas as condições de validação e de teste cego.
- O localizador adaptativo usa escala única acima de 50 dB e consenso de quatro bandas abaixo disso. O pipeline foi congelado antes do teste em `FINAL_PIPELINE_FREEZE.json`.
- Teste cego ideal: cobertura `100%`, MAE `0,665 km`, mediana `0,358 km`, P95 `2,834 km`, máximo `4,368 km`; todas as metas atingidas.
- Teste cego em 60 dB: cobertura `92,86%`, MAE `0,702 km`, P95 `2,960 km`, máximo `4,218 km`; metas atingidas entre conclusivos.
- Ganho `±1%` e sincronização `±1 µs`: cobertura `100%`, com métricas dentro das metas.
- Teste cego em 40 dB: cobertura `47,14%`, mas uma reflexão falsa elevou o máximo para `154,23 km`; reprovado.
- Teste cego em 30 dB isolado: cobertura `27,14%`, MAE `0,275 km`, máximo `1,452 km` entre conclusivos.
- Teste combinado 30 dB+ganho+sync: cobertura `35,71%`, com máximo `169,10 km`; reprovado.
- Como o teste cego já foi aberto, nenhum parâmetro foi reajustado. Foi adicionada apenas uma guarda operacional explícita em `infer_fault.py`: abaixo de 50 dB, classificar normalmente, mas retornar localização inconclusiva. Essa guarda pós-teste não possui avaliação cega própria.
- A faixa efetivamente coberta pelo teste cego foi 15–450 km. Um smoke test posterior no caso conhecido de 1 km retornou 15,84 km; logo, reflexões muito próximas do terminal não são resolvidas pela assinatura atual de 40 µs. A classificação permanece correta, mas a localização de 1–<15 km exige pesquisa adicional e uma nova campanha cega.
