# Descobertas — SINDy, Koopman e reconstrução de estado

**Data:** 17/09/2026

## O que estamos tentando resolver

A baseline usa o sistema de Duffing:

```text
dx1/dt = x2
dx2/dt = -2*x2 - x1*cos(x1 + x2) + u
```

Primeiro foi feita identificação com estado completo.

Depois foi criada uma versão entrada-saída, medindo apenas:

```text
y = x2
```

e usando atrasos de `y` e `u` no lifting de Koopman.

O problema apareceu aqui: o Koopman entrada-saída consegue criar um estado interno útil para previsão, mas esse estado não precisa ter correspondência direta com o estado físico oculto `x1`.

Ou seja:

> prever bem a saída não significa necessariamente reconstruir corretamente o estado físico interno.

---

## Mudança de direção

A ideia passou a ser separar duas coisas:

- **a física conhecida define o significado do estado**;
- **os dados identificam a parte desconhecida da dinâmica**.

No Duffing:

- saída medida: `y = x2`
- estado oculto: `x1`

A única física fornecida ao identificador é:

```text
dx1/dt = y
```

A segunda equação não é fornecida.

Como:

```text
dx1/dt = y
```

então:

```text
x1(t) = x1(0) + integral(y dt)
```

Isso significa que a forma temporal de `x1` pode ser reconstruída a partir de `y`, mas ainda existe uma ambiguidade no valor inicial `x1(0)`.

Na prática, o principal problema passa a ser encontrar esse offset inicial.

---

## Por que usamos SINDy

SINDy foi usado para aprender a parte desconhecida da dinâmica.

Uma biblioteca simples pode ser:

```text
Theta = [
    1,
    x1,
    y,
    u,
    cos(x1 + y),
    x1*cos(x1 + y)
]
```

O modelo é escrito como:

```text
dy/dt = Theta * xi
```

A biblioteca pode conter funções não lineares, mas a regressão continua linear nos coeficientes `xi`.

Essa é a ideia principal do SINDy:

- colocar possíveis não linearidades na biblioteca;
- estimar os coeficientes por regressão;
- eliminar termos que não parecem relevantes.

---

## Como funciona o STLSQ

Foi usada uma implementação simples de **Sequential Thresholded Least Squares (STLSQ)**.

O processo é:

1. fazer mínimos quadrados usando toda a biblioteca;
2. remover coeficientes menores que um threshold;
3. refazer os mínimos quadrados somente com os termos restantes;
4. repetir até estabilizar.

Objetivo:

- manter poucos termos;
- evitar uma equação cheia de coeficientes pequenos;
- obter uma dinâmica mais simples e interpretável.

---

## Primeiro teste — sem ruído

Para cada candidato de `x1(0)`:

1. `x1` foi reconstruído pela integral de `y`;
2. o SINDy identificou a dinâmica de `y`;
3. foi calculado o erro do modelo;
4. o candidato com menor erro foi escolhido.

### Resultado

Sem ruído:

- o valor correto de `x1(0)` foi encontrado;
- a trajetória estimada de `x1` coincidiu praticamente com a real;
- o estado verdadeiro `x1` não foi usado na identificação, apenas na validação.

Esse teste mostrou que a ideia funciona em um caso ideal quando:

- a física parcial está correta;
- os dados estão limpos;
- a biblioteca contém uma estrutura adequada.

---

## Testes com perturbações

Depois foram adicionados:

- ruído de medição de 1%, 3%, 5% e 10%;
- perturbação de processo na planta;
- parâmetros físicos não informados ao SINDy;
- duas bibliotecas diferentes.

### Structured library

Contém o termo verdadeiro:

```text
x1*cos(x1 + y)
```

A intenção é testar o caso em que temos uma boa hipótese sobre a estrutura da dinâmica.

### Blind library

Não contém o termo verdadeiro.

A intenção é verificar o que acontece quando a biblioteca está incompleta e o SINDy precisa aproximar a dinâmica usando termos errados ou apenas parcialmente adequados.

---

## O que funcionou

Mesmo com perturbações, a reconstrução de `x1` frequentemente manteve a forma temporal correta.

Isso acontece porque:

```text
x1(t) = x1(0) + integral(y dt)
```

A integração preserva bem a evolução relativa de `x1`.

Também foi possível obter modelos SINDy que reproduziam razoavelmente `x2` em rollout, mesmo com ruído.

Isso mostrou que o método ainda consegue capturar parte importante da dinâmica observável.

---

## O que começou a quebrar

O maior problema foi a identificação de `x1(0)`.

Em muitos casos:

- a forma de `x1` estava correta;
- mas toda a trajetória estava deslocada verticalmente;
- o RMSE de `x1` ficava próximo do erro no offset inicial.

Isso mostrou que a física conhecida determina bem a variação de `x1`, mas não fixa sua origem.

---

## Sensibilidade ao ruído

SINDy mostrou alta sensibilidade ao ruído principalmente por causa da estimação de `dy/dt`.

A forma mais simples usada foi:

```text
dy/dt ≈ (y[k+1] - y[k]) / dt
```

Com:

```text
dt = 0.01
```

o ruído é amplificado aproximadamente por um fator de:

```text
1/dt = 100
```

A integração usada para reconstruir `x1` tem comportamento oposto e tende a suavizar ruído.

Por isso apareceu uma situação recorrente:

- a forma de `x1` parecia boa;
- o SINDy escolhia um `x1(0)` errado;
- a dinâmica identificada ainda conseguia explicar parte da saída.

---

## Compensação do erro pelo próprio SINDy

Esse foi um dos principais achados.

O SINDy consegue compensar uma realização ruim do estado ajustando outros termos e coeficientes da equação.

Na structured library, mesmo com `x1(0)` errado, o método conseguiu encontrar combinações de termos que explicavam razoavelmente a saída.

Na blind library isso ficou ainda mais claro:

- apareceram coeficientes muito grandes;
- vários termos passaram a ser usados simultaneamente;
- a regressão tentou compensar ruído, biblioteca incompleta, estado deslocado e perturbação de processo ao mesmo tempo.

### Conclusão

> um bom ajuste de saída não garante que o estado físico interno esteja correto.

---

## Rollout do modelo identificado

Foi adicionada a simulação do modelo SINDy encontrado.

Os gráficos passaram a comparar:

- `x2` real;
- medição ruidosa `y`;
- saída do modelo SINDy.

Em alguns casos o rollout de `x2` permaneceu razoável mesmo quando `x1` estava claramente errado.

Isso separou duas perguntas:

1. o modelo consegue reproduzir a saída?
2. o modelo recupera corretamente o estado físico?

Esses dois objetivos não são equivalentes.

---

## Tentativas de melhorar x1(0)

Foram testadas duas mudanças simples:

- diferença central para estimar `dy/dt`;
- penalização de parcimônia na escolha de `x1(0)`.

O score passou a considerar:

```text
erro de ajuste + penalização da complexidade dos coeficientes
```

A ideia foi evitar que um `x1(0)` ruim fosse compensado por uma equação excessivamente complexa.

### Resultado

A análise ficou melhor, mas a ambiguidade do estado inicial não foi eliminada.

---

## Principal conclusão

A dificuldade não é apenas o ruído.

Existe um problema de **identificabilidade da realização física**.

A relação:

```text
dx1/dt = y
```

determina `x1` apenas até uma constante.

O SINDy pode ajustar os coeficientes da dinâmica para acomodar diferentes valores desse offset, principalmente quando:

- há ruído;
- a biblioteca é muito flexível;
- a biblioteca está incompleta.

Por isso:

> reconstruir um estado físico oculto exige mais do que reproduzir bem a entrada-saída.

---

## Próximos passos

Ainda não implementados:

- escolher `x1(0)` pelo erro de rollout da saída, e não apenas pelo erro local em `dy/dt`;
- melhorar a estimação de derivadas;
- testar **weak / integral SINDy** para evitar diferenciação direta de sinais ruidosos;
- usar várias trajetórias com diferentes estados iniciais e uma única dinâmica comum;
- testar calibrações físicas esparsas de `x1` para quebrar a ambiguidade de offset;
- usar os termos encontrados pelo SINDy para orientar depois um lifting Koopman mais interpretável e orientado a controle.

---

## Resumo geral

O caminho construído até aqui foi:

```text
entrada-saída
    ↓
física parcial
    ↓
reconstrução de estado candidato
    ↓
SINDy
    ↓
modelo não linear identificado
```

O principal aprendizado até agora foi separar claramente:

- **previsão de saída**
- **realização física do estado**

Um modelo pode explicar muito bem a saída e ainda assim representar incorretamente o estado físico oculto.
