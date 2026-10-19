# Reunião de pesquisa com Prof. Luciano Frezzato — 19/10/2026

## Anotações registradas durante a reunião

### 1. Experimentos de identificação com estados ocultos
- Investigar como o sistema/modelo se comporta quando **um estado não é disponibilizado ao identificador**.
- Tentar **reconstruir o estado oculto** a partir das medidas disponíveis, inspirado na abordagem previamente experimentada com **SINDy**.
- Considerar, ao planejar o experimento, o benchmark Duffing modificado já presente neste repositório e os experimentos de identificação entrada-saída e port-Hamiltoniana com posição inicial oculta.
- **A investigar (não é resultado):** desempenho de reconstrução, identificabilidade, propagação de erros, sensibilidade a ruído e impacto na previsão/rollout. Decidir o protocolo com o orientador.

### 2. Leituras indicadas pelo professor — pendentes das fotos
- Ler os **artigos 1 e 2 da primeira foto**. A foto com a relação de artigos ainda será enviada; **não presumir títulos, autores ou correspondências**.
- Olhar os **artigos 3 e 5 das fotos enviadas pelo professor**. Fotos ainda serão encaminhadas na conversa; **não presumir títulos nem conteúdo**.
- Quando receber as fotos, identificar as referências, registrar os metadados e estabelecer a ordem de leitura.

### 3. Expansão da linha de modelagem
- **Estudar Gaussian Processes (GPs)**: fundamentos, uso para identificação dinâmica, incerteza e limitações computacionais.
- **Avançar para redes neurais**, após a etapa inicial de métodos clássicos/estruturados, conforme orientação discutida nesta reunião.
- Relacionar essas alternativas com Koopman, SINDy, reconstrução de estados e identificação orientada a controle. Isso é direção de estudo, não resultado experimental.

### 4. Pendências técnicas já levantadas na reunião
- Padronizar a notação de derivadas, explicitando em relação a que variável se está derivando (tempo, coordenadas de estado, parâmetros).
- Conferir **o sinal de uma expressão no final da página 7** do material em leitura. A equação exata ainda precisa ser identificada antes de concluir se há erro de sinal.

### 5. Referência de localização, separada da pesquisa
- **Rua Alvarenga, São Paulo — região do Butantã**. Referência registrada, sem endereço/número específico ou finalidade confirmada.
- A imagem de mapa enviada mostra a região do Butantã, próxima à Av. Vital Brasil e R. Camargo, mas **não comprova que o marcador esteja na Rua Alvarenga**.

## Próximo checkpoint
1. Receber as fotos com a lista de artigos e identificar as referências 1, 2, 3 e 5.
2. Definir um experimento de ocultação/reconstrução de estado com medições e métricas comparáveis às usadas em SINDy.
3. Traçar plano de estudo incremental de GPs e redes neurais.
4. Revisitar o sinal no final da página 7 quando a expressão estiver localizada.

> Registro de reunião: itens de estudo e hipóteses, não conclusões comprovadas. Nenhum experimento foi executado neste commit.
