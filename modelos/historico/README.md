# Modelos e congelamentos históricos

Classificadores treinados (`.joblib`) e documentos de congelamento
(`FINAL_PIPELINE_FREEZE_*.json`) de versões anteriores à geração G2, mantidos
para rastreabilidade — cada congelamento documenta os dados de treino, os
números de validação da época, e o motivo da mudança em relação ao anterior.

**Nenhum artefato desta pasta está em produção.** A versão ativa é a G2.2,
descrita em [`modelos/FINAL_PIPELINE_FREEZE_G2.2.json`](../FINAL_PIPELINE_FREEZE_G2.2.json)
e apontada por [`modelos/classificador_config.master.json`](../classificador_config.master.json).

Ordem cronológica aproximada: `V2` → `V3` → ... → `V29` (numeração sequencial,
descontinuada por ficar confusa com 31 versões) → `G2.0` → `G2.1` → **`G2.2`**
(geração atual, em `modelos/`). A partir da G2.0 o projeto passou a usar
"gerações" (G1, G2.0, G2.1, G2.2...) em vez de números sequenciais, reservadas
para promoções reais a produção — experimentos de pesquisa usam nomes
descritivos (ver `src/legado_experimentos/`).
