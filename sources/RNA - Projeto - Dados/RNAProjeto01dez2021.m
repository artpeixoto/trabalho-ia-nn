clear
clc
%% Estima��o de energia consumida
% Este exemplo de projeto ilustra como uma Rede Neural Artificial (RNA)
% de ajuste de fun��o pode ser utilizada para estimar a energia absorvida
% em um processador de imagens de resson�ncia magn�tica

% Fonte: 
% SILVA, I. N.; SPATTI, D. H.; FLAUZINO, R. A.; LIBONI, L. H. B.; 
% ALVES, S. F. R.. Artificial Neural Networks - A Practical Course.
% Springer, 2017.

%% O problema: Predi��o (estima��o)
% Neste projeto � constru�da RNA que pode estimar
% a energia absorvida em um processador de imagens de resson�ncia magn�tica
% descrita por tr�s atributos f�sicos
% x1, x2 e x3
%
% Este � um exemplo de um problema de ajuste de fun��es, onde as entradas
% s�o combinadas at� que atingir sa�das alvo associadas, sa�da "y".
% O objetivo � criar uma RNA que n�o apenas estima as metas conhecidas
% dadas as entradas conhecidas, mas que possa generalizar e estimar com
% precis�o as sa�das para as entradas que n�o foram usadas para projetar a solu��o.

% A RNA ser� projetada usando as tr�s vari�veis de entrada (atributos)
% cujo valor estimado de energia absorvida em um processador de imagens
% de resson�ncia magn�tica que j� � conhecida (treinamento supervisionado)
% O treinamento � realizado para produzir as avalia��es alvo (sa�da desejada)

%% Preparando os dados de treinamento
% Os dados para problemas de ajuste de fun��o s�o configurados para uma RNA
% s�o organizados em duas matrizes, a matriz de entrada "X" e a matriz alvo
% "T".
% Cada i-�sima coluna da matriz de entrada ter� tr�s elementos representando
% cada atributo de entrada conhecido.
%
% Cada coluna correspondente da matriz de destino ter� um elemento, 
% representando o valor estimado de energia absorvida no processador
% de imagens de resson�ncia magn�tica, que j� � conhecida

%% Carregamento dos Dados de Treinamento - Atributos de Entrada
% 200 amostras
% 3 atributos (x1, x2 e x3) de entrada para cada amostra
% Base de dados - Planilha Excel

nome_arquivo = 'DadosProjeto01RNA.xlsx';
sheet = 'DadosTreinamentoRNA';

AtributosEntrada = xlsread(nome_arquivo,sheet,'B2:D201');
save('AtributosEntrada.mat','AtributosEntrada');

%% Carregamento dos Dados de Treinamento - Sa�da Desejada
% 200 amostras
% 1 sa�da desejada ( d )para cada amostra
% Base de dados - Planilha Excel

nome_arquivo = 'DadosProjeto01RNA.xlsx';
sheet = 'DadosTreinamentoRNA';

SaidaDesejada = xlsread(nome_arquivo,sheet,'E2:E201');
save('SaidaDesejada.mat','SaidaDesejada');

X = AtributosEntrada';
T = SaidaDesejada';

%% Carregamento dos Dados de Teste (aplica��o) - Atributos de Entrada
% 20 amostras
% 3 atributos (x1, x2 e x3) de entrada para cada amostra
% Base de dados - Planilha Excel

nome_arquivo = 'DadosProjeto01RNA.xlsx';
sheet = 'DadosTesteRNA';

AtributosEntradaTeste = xlsread(nome_arquivo,sheet,'B2:D21');
save('AtributosEntradaTeste.mat','AtributosEntradaTeste');

%% Carregamento dos Dados de Teste (aplica��o) - Sa�da Desejada (alvo)
% 20 amostras
% 1 sa�da desejada ( d )para cada amostra
% Base de dados - Planilha Excel

nome_arquivo = 'DadosProjeto01RNA.xlsx';
sheet = 'DadosTesteRNA';

SaidaDesejadaTeste = xlsread(nome_arquivo,sheet,'E2:E21');
save('SaidaDesejadaTeste.mat','SaidaDesejadaTeste');

testeX = AtributosEntradaTeste';
testeT = SaidaDesejadaTeste';

%%
% � poss�vel visualizar as dimens�es das entradas "X" e alvos "T".
% Observar que "X" e "T" t�m 200 colunas. Estes representam 200 entradas
% (amostras de treinamento) e 200 sa�das associadas (metas).
% % A matriz de entrada "X" tem tr�s linhas, para os tr�s atributos.
% A matriz de destino "T" tem apenas uma linha, pois para cada amostra de
% de treinamento temos apenas uma sa�da desejada, o valor estimado de
% energia absorvida no processador de imagens de resson�ncia magn�tica.

size(X)
size(T)

%% Ajustando uma fun��o com uma RNA
% O pr�ximo passo � criar uma rede neural que aprender� a estimar 
% o valor de % energia absorvida no processador de imagens de resson�ncia magn�tica.
%
% Uma vez que a RNA come�a com pesos iniciais aleat�rios, os resultados
% deste projeto ser� ligeiramente diferente toda vez que for executado. 
% A "semente" aleat�ria (valores iniciais dos pesos das sin�pses) est�
% definido para evitar esta aleatoriedade.
% No entanto, isso n�o � necess�rio para suas pr�prias aplica��es.

setdemorandstream(491218382)

%%
% Redes Neurais Artificiais de "feed forward" de duas camadas
%(ou seja, uma camada oculta) podem atender qualquer rela��o de
% entrada-sa�da com neur�nios suficientes na camada oculta.

% Camadas que n�o s�o camadas de sa�da s�o chamadas de camadas ocultas.

% Neste projeto ser� utilizada uma �nica camada oculta de 10 neur�nios.

% Problemas gerais, mais dif�ceis requerem mais neur�nios, e talvez mais
% camadas. Problemas mais simples requerem menos neur�nios.
%
% A entrada e sa�da t�m tamanhos de 0 porque a rede ainda n�o foi
% configurada para corresponder aos nossos dados de entrada e de destino.
% Isso vai acontecer quando a RNA estiver treinada.

net = fitnet(10);
view(net)

%%  TREINANDO a Rede Neural Artificial
% Agora a RNA est� pronta para ser treinada.
% As "amostras de treinamento (no caso 200 amostras)" s�o automaticamente
% divididas em conjuntos de treinamento, valida��o e teste. 
% O conjunto de treinamento � usado para ensinar a RNA.
% O treinamento continua enquanto a RNA continuar melhorando no conjunto
% de valida��o. 
% O conjunto de teste fornece uma completa medida independente da precis�o RNA.
%
% A ferramenta de treinamento de rede neural mostra a rede sendo treinada
% e os algoritmos comumente utilizados para trein�-la. 
% Ela tamb�m exibe o estado do treinamento durante o treinamento e
% os crit�rios que interromperam o treinamento ser�o destacados em verde.
%
% Os bot�es na parte inferior abrem gr�ficos �teis que podem ser
% visualizados durante e ap�s o treinamento. 

[net,tr] = train(net,X,T);
nntraintool

%% Verificando o DESEMPENHO DO TREINAMENTO da RNA
% Para ver como o desempenho da rede melhorou durante o treinamento,
% � s� clicar no bot�o "Performance" na ferramenta de treinamento ou
% execute o comando PLOTPERFORM.
%
% O desempenho do treinamento da RNA � medido em termos de
% Erro Quadr�tico M�dio e mostrado em escala logar�tmica. 
% Este erro diminuiu rapidamente conforme a RNA foi treinada.
% O desempenho da RNA � mostrado para cada um dos conjuntos de treinamento,
% valida��o e teste.
% A RNA final � a rede com melhor desempenho no processo de valida��o.

plotperform(tr)

%% TESTANDO a RNA  (processo de aplica��o da RNA) 
% O Erro Quadr�tico m�dio da RNA treinada agora pode ser medido
% em rela��o �s amostras de teste (aplica��o).
% Este processo permite verificar se a RNA funcionar� bem (apresentar� bons
% resultados) quando aplicada a dados do mundo real.

testeY = net(testeX);

DesempenhoEQM = mse(net,testeT,testeY)

VarianciaTesteY=var(testeY)

%%
% Uma outra segunda medida de qu�o bem a rede neural ajustou os dados � o gr�fico de
% regress�o. Aqui, a regress�o � tra�ada em todas as amostras.
%
% O gr�fico de regress�o mostra as sa�das de rede reais plotadas em termos
% dos valores alvo associados. Se a rede aprendeu a ajustar bem os dados,
% o ajuste linear para esta rela��o de sa�da-destino deve estreitamente
% interceptar os cantos inferior esquerdo e superior direito do gr�fico.
%
% Se este n�o for o caso, ent�o um treinamento adicional ou treinamento
% de uma rede com mais neur�nios ocultos, seria aconselh�vel.

Y = net(X);

plotregression(T,Y)

%%
% Uma outra terceira medida de qu�o bem a rede neural ajustou os dados � o
% histograma de erro. Ele mostra como os tamanhos dos erros s�o distribu�dos.
% Normalmente, a maioria dos erros est� perto de zero, com muito poucos
% erros longe disso.

e =testeT -testeY;
ploterrhist(e)

ErroRelativoMedio=sum(e./testeT)/20
