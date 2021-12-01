clear
clc
%% Estimação de energia consumida
% Este exemplo de projeto ilustra como uma Rede Neural Artificial (RNA)
% de ajuste de função pode ser utilizada para estimar a energia absorvida
% em um processador de imagens de ressonância magnética

% Fonte: 
% SILVA, I. N.; SPATTI, D. H.; FLAUZINO, R. A.; LIBONI, L. H. B.; 
% ALVES, S. F. R.. Artificial Neural Networks - A Practical Course.
% Springer, 2017.

%% O problema: Predição (estimação)
% Neste projeto é construída RNA que pode estimar
% a energia absorvida em um processador de imagens de ressonância magnética
% descrita por três atributos físicos
% x1, x2 e x3
%
% Este é um exemplo de um problema de ajuste de funções, onde as entradas
% são combinadas até que atingir saídas alvo associadas, saída "y".
% O objetivo é criar uma RNA que não apenas estima as metas conhecidas
% dadas as entradas conhecidas, mas que possa generalizar e estimar com
% precisão as saídas para as entradas que não foram usadas para projetar a solução.

% A RNA será projetada usando as três variáveis de entrada (atributos)
% cujo valor estimado de energia absorvida em um processador de imagens
% de ressonância magnética que já é conhecida (treinamento supervisionado)
% O treinamento é realizado para produzir as avaliações alvo (saída desejada)

%% Preparando os dados de treinamento
% Os dados para problemas de ajuste de função são configurados para uma RNA
% são organizados em duas matrizes, a matriz de entrada "X" e a matriz alvo
% "T".
% Cada i-ésima coluna da matriz de entrada terá três elementos representando
% cada atributo de entrada conhecido.
%
% Cada coluna correspondente da matriz de destino terá um elemento, 
% representando o valor estimado de energia absorvida no processador
% de imagens de ressonância magnética, que já é conhecida

%% Carregamento dos Dados de Treinamento - Atributos de Entrada
% 200 amostras
% 3 atributos (x1, x2 e x3) de entrada para cada amostra
% Base de dados - Planilha Excel

nome_arquivo = 'DadosProjeto01RNA.xlsx';
sheet = 'DadosTreinamentoRNA';

AtributosEntrada = xlsread(nome_arquivo,sheet,'B2:D201');
save('AtributosEntrada.mat','AtributosEntrada');

%% Carregamento dos Dados de Treinamento - Saída Desejada
% 200 amostras
% 1 saída desejada ( d )para cada amostra
% Base de dados - Planilha Excel

nome_arquivo = 'DadosProjeto01RNA.xlsx';
sheet = 'DadosTreinamentoRNA';

SaidaDesejada = xlsread(nome_arquivo,sheet,'E2:E201');
save('SaidaDesejada.mat','SaidaDesejada');

X = AtributosEntrada';
T = SaidaDesejada';

%% Carregamento dos Dados de Teste (aplicação) - Atributos de Entrada
% 20 amostras
% 3 atributos (x1, x2 e x3) de entrada para cada amostra
% Base de dados - Planilha Excel

nome_arquivo = 'DadosProjeto01RNA.xlsx';
sheet = 'DadosTesteRNA';

AtributosEntradaTeste = xlsread(nome_arquivo,sheet,'B2:D21');
save('AtributosEntradaTeste.mat','AtributosEntradaTeste');

%% Carregamento dos Dados de Teste (aplicação) - Saída Desejada (alvo)
% 20 amostras
% 1 saída desejada ( d )para cada amostra
% Base de dados - Planilha Excel

nome_arquivo = 'DadosProjeto01RNA.xlsx';
sheet = 'DadosTesteRNA';

SaidaDesejadaTeste = xlsread(nome_arquivo,sheet,'E2:E21');
save('SaidaDesejadaTeste.mat','SaidaDesejadaTeste');

testeX = AtributosEntradaTeste';
testeT = SaidaDesejadaTeste';

%%
% É possível visualizar as dimensões das entradas "X" e alvos "T".
% Observar que "X" e "T" têm 200 colunas. Estes representam 200 entradas
% (amostras de treinamento) e 200 saídas associadas (metas).
% % A matriz de entrada "X" tem três linhas, para os três atributos.
% A matriz de destino "T" tem apenas uma linha, pois para cada amostra de
% de treinamento temos apenas uma saída desejada, o valor estimado de
% energia absorvida no processador de imagens de ressonância magnética.

size(X)
size(T)

%% Ajustando uma função com uma RNA
% O próximo passo é criar uma rede neural que aprenderá a estimar 
% o valor de % energia absorvida no processador de imagens de ressonância magnética.
%
% Uma vez que a RNA começa com pesos iniciais aleatórios, os resultados
% deste projeto será ligeiramente diferente toda vez que for executado. 
% A "semente" aleatória (valores iniciais dos pesos das sinápses) está
% definido para evitar esta aleatoriedade.
% No entanto, isso não é necessário para suas próprias aplicações.

setdemorandstream(491218382)

%%
% Redes Neurais Artificiais de "feed forward" de duas camadas
%(ou seja, uma camada oculta) podem atender qualquer relação de
% entrada-saída com neurônios suficientes na camada oculta.

% Camadas que não são camadas de saída são chamadas de camadas ocultas.

% Neste projeto será utilizada uma única camada oculta de 10 neurônios.

% Problemas gerais, mais difíceis requerem mais neurônios, e talvez mais
% camadas. Problemas mais simples requerem menos neurônios.
%
% A entrada e saída têm tamanhos de 0 porque a rede ainda não foi
% configurada para corresponder aos nossos dados de entrada e de destino.
% Isso vai acontecer quando a RNA estiver treinada.

net = fitnet(10);
view(net)

%%  TREINANDO a Rede Neural Artificial
% Agora a RNA está pronta para ser treinada.
% As "amostras de treinamento (no caso 200 amostras)" são automaticamente
% divididas em conjuntos de treinamento, validação e teste. 
% O conjunto de treinamento é usado para ensinar a RNA.
% O treinamento continua enquanto a RNA continuar melhorando no conjunto
% de validação. 
% O conjunto de teste fornece uma completa medida independente da precisão RNA.
%
% A ferramenta de treinamento de rede neural mostra a rede sendo treinada
% e os algoritmos comumente utilizados para treiná-la. 
% Ela também exibe o estado do treinamento durante o treinamento e
% os critérios que interromperam o treinamento serão destacados em verde.
%
% Os botões na parte inferior abrem gráficos úteis que podem ser
% visualizados durante e após o treinamento. 

[net,tr] = train(net,X,T);
nntraintool

%% Verificando o DESEMPENHO DO TREINAMENTO da RNA
% Para ver como o desempenho da rede melhorou durante o treinamento,
% é só clicar no botão "Performance" na ferramenta de treinamento ou
% execute o comando PLOTPERFORM.
%
% O desempenho do treinamento da RNA é medido em termos de
% Erro Quadrático Médio e mostrado em escala logarítmica. 
% Este erro diminuiu rapidamente conforme a RNA foi treinada.
% O desempenho da RNA é mostrado para cada um dos conjuntos de treinamento,
% validação e teste.
% A RNA final é a rede com melhor desempenho no processo de validação.

plotperform(tr)

%% TESTANDO a RNA  (processo de aplicação da RNA) 
% O Erro Quadrático médio da RNA treinada agora pode ser medido
% em relação às amostras de teste (aplicação).
% Este processo permite verificar se a RNA funcionará bem (apresentará bons
% resultados) quando aplicada a dados do mundo real.

testeY = net(testeX);

DesempenhoEQM = mse(net,testeT,testeY)

VarianciaTesteY=var(testeY)

%%
% Uma outra segunda medida de quão bem a rede neural ajustou os dados é o gráfico de
% regressão. Aqui, a regressão é traçada em todas as amostras.
%
% O gráfico de regressão mostra as saídas de rede reais plotadas em termos
% dos valores alvo associados. Se a rede aprendeu a ajustar bem os dados,
% o ajuste linear para esta relação de saída-destino deve estreitamente
% interceptar os cantos inferior esquerdo e superior direito do gráfico.
%
% Se este não for o caso, então um treinamento adicional ou treinamento
% de uma rede com mais neurônios ocultos, seria aconselhável.

Y = net(X);

plotregression(T,Y)

%%
% Uma outra terceira medida de quão bem a rede neural ajustou os dados é o
% histograma de erro. Ele mostra como os tamanhos dos erros são distribuídos.
% Normalmente, a maioria dos erros está perto de zero, com muito poucos
% erros longe disso.

e =testeT -testeY;
ploterrhist(e)

ErroRelativoMedio=sum(e./testeT)/20
