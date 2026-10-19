# Duffing Port-Hamiltoniano — identificação com ruído

## Objetivo e referência

Estudar a robustez do identificador PH-1 paramétrico do arquivo port_hamiltonian_duffing.py, reutilizando a convenção de SINDy_with_disturbances.py: ruído de medição de 1%, 3%, 5% e 10%, além de perturbação de processo com desvio-padrão 0,5. O teste adicional de 0% é apenas diagnóstico.

A planta de referência é:

\[
\dot q=p,\qquad \dot p=-2p-q\cos(q+p)+u+w,\qquad y=p+v.
\]

O modelo PH mantém J canônico e G=[0,1]^T, aprendendo três coeficientes positivos (a, r0, r1) de:

\[
H_a(q,p)=p^2/2+a[q\sin q+\cos q-1],
\]

\[
R(q,p)=\operatorname{diag}(0,r_0+r_1[1-q\sin(q+p/2)\operatorname{sinc}(p/2)]).
\]

O valor verdadeiro é a=r0=r1=1. A identidade trigonométrica 2 sin(p/2) sin(q+p/2)=cos(q)-cos(q+p) demonstra que a representação de referência reproduz exatamente a planta não perturbada. Para |q|<=1 e coeficientes não negativos, R é semidefinida positiva. Esta é uma **garantia local de estrutura**, não uma prova de estabilidade assintótica.

## Protocolo reproduzível

- Semente 156, dt=0,01 s, integração RK4, 10 s.
- Treino: x0=(-0,6;1,4), u[k]=0,8 sin(0,2k), uma mesma realização de w[k] ~ N(0;0,5²) para todos os níveis de ruído.
- A única saída observada é y=p+v, com desvio-padrão do ruído v igual ao nível multiplicado por std(p verdadeiro), como no experimento SINDy.
- Para reconstruir q, fornecemos **o valor verdadeiro de q(0)** e integramos a velocidade medida após suavização. Isso favorece PH-1 em relação ao SINDy, que busca o estado inicial oculto.
- Derivadas estimadas dos dados por filtro polinomial local (janela de 41 pontos, grau 3). Não são fornecidas as derivadas verdadeiras da planta.
- Identificação direta de a,r0,r1 por mínimos quadrados não negativos (NNLS). Como há apenas três coeficientes, o código testa os oito subconjuntos ativos e resolve cada um por `torch.linalg.lstsq`.
- O erro minimizado vem dos dados: `Y = p_dot_estimado - u` e `Phi = Phi(q_estimado, p_estimado)`.
- Validação limpa e fora da trajetória de treino: x0=(0,4;-1,0), u[k]=0,65 sin(0,17k+0,4); métricas de predição de um passo e rollout livre em 10 s.

## Resultados e interpretação

**Nota de versão:** as métricas da execução anterior foram obtidas com Adam,
antes da refatoração para mínimos quadrados. A tabela numérica antiga foi
retirada para não atribuir aqueles resultados ao novo identificador.
Execute o script para obter os novos coeficientes e métricas.

O script imprime, para cada nível de ruído, os coeficientes identificados,
MSE do ajuste das derivadas, erro na reconstrução de posição, erro de
predição de um passo, erro de rollout e menor dissipação observada.

Observações metodológicas:

1. **Bom ajuste preditivo não significa recuperar parâmetros exatos.**
   Os parâmetros podem não ser individualmente identificáveis a partir dos
   dados disponíveis, mesmo quando o rollout é bom.
2. **Mesmo com 0% de ruído de medição**, a perturbação de processo e a
   suavização podem introduzir erros na regressão.
3. **Uma realização aleatória por nível de ruído não demonstra robustez
   estatística.** Para isso, repetir o estudo com várias sementes.
4. **A comparação com Koopman/SINDy requer hipóteses equivalentes:**
   este PH recebe a forma exata de H e R e, no experimento ruidoso,
   também conhece q(0).
5. **A restrição de dissipação é local:** os coeficientes são não negativos,
   mas R >= 0 só é assegurado pela parametrização quando |q| <= 1.

## Como executar

A partir da raiz do repositório, após instalar requirements.txt:

    python systems/duffing_oscillator/port_hamiltonian_duffing.py
    python systems/duffing_oscillator/port_hamiltonian_duffing_with_disturbances.py

Para também estimar a posição inicial desconhecida, execute:

    python systems/duffing_oscillator/port_hamiltonian_duffing_hidden_state.py

Os três scripts agora compartilham a mesma regressão port-Hamiltoniana e a
mesma rotina de mínimos quadrados, sem Adam. O próximo passo conceitual é
aprender H_theta e R_theta com parametrizações mais flexíveis, preservando as
restrições estruturais e comparando em igualdade de informação.
