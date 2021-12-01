clear
clc
%% Carregamento Dados Treinamento - Atributos de Entrada
%  200 amostras
%  3 atributos (x1, x2 e x3) de entrada para cada amostra

nome_arquivo = 'DadosProjeto01RNA.xlsx';
sheet = 'DadosFormatadosRNA';

AtributosEntrada = xlsread(nome_arquivo,sheet,'B2:D201');
save('AtributosEntrada.mat','AtributosEntrada');

%% Carregamento Dados Treinamento - Saída Desejada
%  200 amostras
%  1 saída desejada ( d )para cada amostra

nome_arquivo = 'DadosProjeto01RNA.xlsx';
sheet = 'DadosFormatadosRNA';

SaidaDesejada = xlsread(nome_arquivo,sheet,'E2:E201');
save('SaidaDesejada.mat','SaidaDesejada');