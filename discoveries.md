# Descobertas — identificação entrada-saída, SINDy e reconstrução de estado

**Data:** 17/09/2026

## Contexto

A linha de trabalho começou a partir da baseline de Koopman no Duffing:

\[
\dot{x}_1 = x_2
\]

\[
\dot{x}_2 = -2x_2 - x_1\cos(x_1+x_2) + u
\]

Primeiro foi feita identificação com estado completo. Depois foi criada uma versão entrada-saída, medindo apenas

\[
y = x_2
\]

e usando atrasos de \(y\) e \(u\) no lifting de Koopman.

## Limitação percebida no Koopman entrada-saída

O modelo entrada-saída consegue prever a dinâmica usando estados construídos a partir de atrasos, mas essas coordenadas não têm necessariamente correspondência com os estados físicos internos.

Por isso, aplicar um observador diretamente nesse espaço aumentado não garante a reconstrução de um estado físico específico, como \(x_1\). Uma realização interna pode representar corretamente a entrada-saída sem que nenhuma coordenada seja exatamente o estado físico oculto.

A partir disso, a direção passou a ser:

- usar a física conhecida para definir o significado de algumas coordenadas;
- usar os dados para identificar a parte desconhecida da dinâmica.

## Escolha do experimento no Duffing

Foi mantido:

\[
y = x_2
\]

como saída medida e \(x_1\) como estado oculto.

A única física fornecida ao identificador é:

\[
\dot{x}_1 = y
\]

A segunda equação não é fornecida.

Como

\[
\dot{x}_1 = y
\]

então

\[
x_1(t) = x_1(0) + \int_0^t y(\tau)\,d\tau
\]

Isso significa que a forma temporal de \(x_1\) pode ser reconstruída a partir de \(y\), mas existe uma ambiguidade em \(x_1(0)\). O problema de realização do estado passa a ser, em grande parte, encontrar esse offset inicial.

## Por que SINDy

SINDy foi usado para aprender a parte desconhecida da dinâmica.

A ideia central é montar uma biblioteca de funções candidatas, por exemplo:

\[
\Theta =
\left[
1,\;
x_1,\;
y,\;
u,\;
\cos(x_1+y),\;
x_1\cos(x_1+y)
\right]
\]

e escrever:

\[
\dot{y} = \Theta \xi
\]

As funções da biblioteca podem ser não lineares nos estados, mas a regressão é linear nos coeficientes \(\xi\).

A não linearidade está nas colunas de \(\Theta\); o SINDy procura apenas os coeficientes que combinam essas funções.

## STLSQ

Foi usada uma implementação simples de **Sequential Thresholded Least Squares**.

O processo é:

1. resolver mínimos quadrados usando toda a biblioteca;
2. eliminar coeficientes com módulo abaixo de um threshold;
3. refazer os mínimos quadrados apenas com os termos sobreviventes;
4. repetir até estabilizar.

O objetivo é obter uma equação esparsa e interpretável, evitando muitos termos pequenos usados apenas para melhorar marginalmente o ajuste.

## Primeiro experimento — sem ruído

Para cada candidato de \(x_1(0)\):

1. \(x_1\) foi reconstruído pela integral de \(y\);
2. o SINDy identificou a dinâmica de \(y\);
3. foi calculado o erro do modelo;
4. o candidato com menor erro foi escolhido.

Sem ruído, o método encontrou corretamente \(x_1(0)\) e a trajetória estimada de \(x_1\) coincidiu praticamente com a real.

O estado verdadeiro \(x_1\) não foi usado na identificação, apenas na avaliação final.

Esse primeiro teste mostrou que, com física parcial correta, dados limpos e uma biblioteca contendo a estrutura adequada, é possível recuperar um estado físico oculto sem fornecer sua trajetória ao identificador.

## Experimentos com perturbações

Depois foram adicionados:

- ruído de medição de 1%, 3%, 5% e 10%;
- perturbação de processo na planta;
- parâmetros físicos não informados ao SINDy;
- duas bibliotecas diferentes.

### Structured library

Contém o termo verdadeiro:

\[
x_1\cos(x_1+y)
\]

entre os candidatos.

### Blind library

Não contém o termo verdadeiro e oferece outras funções plausíveis, para verificar como o SINDy se comporta quando a biblioteca está incompleta.

## Achados com ruído

A reconstrução de \(x_1\) frequentemente manteve a forma correta, mas com offset errado.

Isso acontece porque a integração de \(y\) reconstrói bem a variação de \(x_1\), enquanto \(x_1(0)\) continua livre.

Em vários testes, o RMSE de \(x_1\) ficou praticamente igual ao erro em \(x_1(0)\).

Também ficou claro que SINDy é sensível ao ruído quando \(\dot{y}\) é estimado numericamente.

A aproximação simples

\[
\dot{y}_k \approx \frac{y_{k+1}-y_k}{\Delta t}
\]

amplifica ruído por aproximadamente \(1/\Delta t\).

Com

\[
\Delta t = 0{,}01
\]

esse fator é 100.

A integração usada para reconstruir \(x_1\) tem o comportamento oposto e tende a suavizar o ruído. Por isso foi possível observar casos em que a forma de \(x_1\) estava correta, mas o estado inteiro estava deslocado devido à escolha errada de \(x_1(0)\).

## Compensação do ruído pelo modelo

Um achado importante foi que o SINDy consegue compensar uma realização de estado ruim alterando termos e coeficientes da equação.

Com ruído, a structured library chegou a escolher um \(x_1(0)\) incorreto e ainda produzir uma dinâmica que explicava razoavelmente a saída.

Na blind library isso ficou ainda mais evidente: apareceram coeficientes muito grandes em vários termos. A regressão estava tentando compensar simultaneamente:

- ruído;
- biblioteca incompleta;
- estado deslocado;
- perturbação de processo.

Portanto, um bom ajuste de saída não implica que a realização física do estado esteja correta.

## Rollout do modelo identificado

Foi adicionada a simulação do modelo SINDy identificado para comparar:

- \(x_2\) real;
- medição ruidosa \(y\);
- saída do modelo SINDy.

Em alguns casos o rollout de \(x_2\) permaneceu razoável mesmo quando \(x_1\) estava mal reconstruído.

Esse resultado separa duas perguntas diferentes:

1. o modelo consegue reproduzir a saída?
2. o modelo recupera corretamente o estado físico interno?

O primeiro problema pode estar relativamente bem resolvido enquanto o segundo continua ambíguo.

## Tentativas para melhorar \(x_1(0)\)

Foram testadas duas mudanças simples:

- diferença central para estimar \(\dot{y}\);
- penalização de parcimônia na escolha de \(x_1(0)\), usando erro de ajuste mais um termo proporcional à norma \(L_1\) dos coeficientes.

A intenção da penalização foi evitar que um estado inicial ruim fosse compensado por uma equação excessivamente complexa.

Essas mudanças melhoraram a análise, mas não eliminaram a ambiguidade do estado inicial.

## Principal conclusão até agora

A dificuldade não é apenas o ruído.

Existe um problema de **identificabilidade da realização física**.

A relação

\[
\dot{x}_1 = y
\]

determina \(x_1\) apenas até uma constante.

O SINDy pode ajustar seus coeficientes para acomodar diferentes valores dessa constante, principalmente quando há ruído ou quando a biblioteca é flexível ou incompleta.

Assim, reconstruir um estado físico oculto exige mais do que reproduzir bem a entrada-saída.

## Próximas ideias

Ainda não implementadas:

- escolher \(x_1(0)\) pelo erro de rollout da saída, em vez do erro local em \(\dot{y}\);
- melhorar a estimação de derivadas;
- testar weak/integral SINDy para evitar diferenciação direta de sinais ruidosos;
- usar várias trajetórias com diferentes estados iniciais e uma única dinâmica comum;
- testar calibrações físicas esparsas de \(x_1\) para quebrar a ambiguidade de offset;
- usar os termos encontrados pelo SINDy para orientar posteriormente um lifting Koopman mais interpretável e orientado a controle.

## Interpretação geral

O caminho que apareceu até aqui é:

\[
\text{entrada-saída}
\rightarrow
\text{física parcial}
\rightarrow
\text{reconstrução de estado candidato}
\rightarrow
\text{SINDy}
\rightarrow
\text{modelo não linear identificado}
\]

O principal aprendizado foi separar claramente **previsão de saída** de **realização física do estado**. Um modelo pode explicar muito bem a saída e ainda assim representar incorretamente o estado físico oculto.
