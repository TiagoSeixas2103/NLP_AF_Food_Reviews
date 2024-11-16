# NLP_AF_Food_Reviews

## Roteiro

1 - Primeiro nós usamos GLoVe para criar embeddings, onde o glove tem uma base de palavras que nós usamos de vocabulário para então transformar o texto que iremos classificar em vetores.

2 - Depois, processamos a database, deixando apenas dois valores de output (ratings maiores que três são 1, menores são 0, demais desconsiderados)

3 - Separamos a database em segmentos de teste e treinamento, sendo o segmento de treinamento 80% das linhas da nossa database. X_train representa os textos do segmento de treinamento a serem classificados, e y_train as classes.  

4 - Realizamos o processo de tokenização

5 - Criamos os batches para cada epoch

6 - Iteramos o modelo pelo número de epochs, treinando-o e aprimorando-o

7 - Otimizador modifica parametros para melhor o sistema em relação a uma funcao de perdas, que serve para medir a quantidade de erros que ocorrem no treinamento. BCEWithLogitsLoss foi usada por ser uma função de classificação para binários.