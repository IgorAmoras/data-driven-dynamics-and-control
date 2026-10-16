# Duffing Port-Hamiltoniano — identificação com ruído

## Objetivo e referência

Estudar a robustez do identificador PH-1 paramétrico do arquivo port_hamiltonian_duffing_intro.py, reutilizando a convenção de SINDy_with_disturbances.py: ruído de medição de 1%, 3%, 5% e 10%, além de perturbação de processo com desvio-padrão 0,5. O teste adicional de 0% é apenas diagnóstico.

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
- Ajuste por Adam de a,r0,r1, com restrição de positividade via softplus.
- Validação limpa e fora da trajetória de treino: x0=(0,4;-1,0), u[k]=0,65 sin(0,17k+0,4); métricas de predição de um passo e rollout livre em 10 s.

## Resultados de uma execução (semente 156)

| Ruído medição | a | r0 | r1 | RMSE q reconstruído | RMSE um passo (teste) | RMSE rollout (teste) |
|---:|---:|---:|---:|---:|---:|---:|
| 0% (diagnóstico) | 0,8923 | 0,8203 | 1,1644 | 0,00078 | 0,00004188 | 0,002852 |
| 1% | 0,8923 | 0,8211 | 1,1642 | 0,00077 | 0,00004228 | 0,002852 |
| 3% | 0,9088 | 0,8088 | 1,1679 | 0,00134 | 0,00003152 | 0,002505 |
| 5% | 0,8848 | 0,8306 | 1,1738 | 0,00200 | 0,00006065 | 0,003320 |
| 10% | 0,9138 | 0,8205 | 1,1513 | 0,00158 | 0,00002908 | 0,002515 |

O script também imprime o MSE de treinamento, o mínimo observado de R e as amplitudes de q, e gera um gráfico das trajetórias no arquivo duffing_ph_noise_rollouts.png (não versionado).

## Análise

1. **Bom ajuste preditivo não significa parâmetros exatos.** Os rollouts de teste ficaram próximos da planta, mas a,r0,r1 não convergiram individualmente a 1.
2. **Mesmo no caso de 0% de ruído no sensor há viés**, porque persistem a perturbação de processo não observada, a suavização e a excitação limitada. Portanto não podemos atribuir toda a diferença de parâmetros ao sensor.
3. **r0+r1 permanece próximo de 2** embora r0 e r1 variem individualmente, sugerindo que o conjunto de estados observado informa melhor a dissipação combinada do que cada parcela.
4. **A sequência de RMSE não é monotônica** (10% não foi pior que 5% neste teste). Há uma realização aleatória por nível; isso não estabelece curva de robustez estatística. Próximo passo: repetir com várias sementes e reportar média e dispersão.
5. **Não é uma comparação direta contra Koopman ou SINDy**: o PH paramétrico recebe a forma funcional verdadeira de H e R e conhece q(0); os demais métodos não recebem necessariamente as mesmas informações e usam procedimentos diferentes. Uma comparação justa deve igualar dados, hipóteses, testes e métricas.

## Como executar

A partir da raiz do repositório, após instalar requirements.txt:

    python systems/duffing_oscillator/port_hamiltonian_duffing_intro.py
    python systems/duffing_oscillator/port_hamiltonian_duffing_with_disturbances.py

A etapa futura será aprender H_theta e R_theta com parametrizações mais flexíveis, preservando as restrições estruturais e comparando em igualdade de informação.
