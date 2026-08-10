# Relatório final do piloto

Data: 3 de agosto de 2026.

## Resultado entregue

- Leitor PL4 validado para 12 canais.
- Geração reproduzível de 500 simulações ATP/JMarti em dez classes.
- Classificador robusto treinado apenas no treino e congelado por SHA-256.
- Localizador adaptativo por ondas viajantes, com escala única ou consenso de quatro bandas.
- CLI `infer_fault.py` para novos PL4.
- 35 testes automatizados aprovados.
- Arquivo de continuidade: `CONTINUACAO_AGENTE.md`.

## Classificador — teste cego

Macro-F1 e acurácia de 100% nas sete condições: ideal, SNR 60/40/30 dB,
ganho ±1%, sincronização ±1 µs e combinação severa. Para o escopo deste piloto,
o classificador das dez classes está concluído.

## Localizador — teste cego

Na condição ideal, com distâncias cegas entre 15 e 450 km: cobertura 100%, MAE
0,665 km, mediana 0,358 km, P95 2,834 km e máximo 4,368 km. Em 60 dB, ganho
±1% e sincronização ±1 µs, os resultados conclusivos também permaneceram nas
metas.

O localizador não foi aprovado em 40 dB nem no cenário combinado de 30 dB,
pois algumas reflexões falsas foram aceitas. Em 30 dB isolado, os poucos casos
conclusivos foram precisos, mas a cobertura foi apenas 27,14%.

Um teste posterior no caso conhecido de 1 km retornou 15,84 km. Portanto, a
faixa 1–500 km originalmente pretendida não foi integralmente validada; o
resultado cego sustenta somente 15–450 km. Resolver faltas próximas exige uma
assinatura temporal menor e um novo teste cego independente.

## Política operacional

`infer_fault.py` sempre fornece a classificação. A localização é bloqueada
abaixo de SNR estimado de 50 dB e inclui advertência sobre faltas próximas ao
terminal. Essa guarda foi adicionada depois do teste cego e, por isso, não deve
ser apresentada como resultado cego validado.

## Evidências imutáveis

- Congelamento: `FINAL_PIPELINE_FREEZE.json`, SHA-256 `7EC8C7B41E2C5B251BBC43ADA6A365019865F58C07F3CE3FA299166454E670D5`.
- Classificador: SHA-256 `3382B29B487CFE3EA350AC15B47D901A28CFEBD904EF4D90EF2384FA2360A9DF`.
- Relatório cego: `campaign_v4/final_blind_test/report.json`, SHA-256 `20BDEC7CD6E08FDD57CEC088BFF23076EFF8AEC03DACD9B37A2E9DD0909F3771`.

Qualquer ajuste futuro deve usar nova validação e nova campanha cega; os 70
casos atuais não podem ser reutilizados para alegar desempenho imparcial.
