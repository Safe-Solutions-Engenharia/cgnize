"""

A intenção aqui é reunir todas as ferramentas úteis relacionadas à abertura dos .cgns 
e simplificar a utilização ao máximo possível, reunindo neste arquivo como uma biblioteca
do tipo wrapper, facilita a utilização de um determinado pacote de código afunilado ao que
nós queremos.

"""

# Importando as bibliotecas:
import os
import logging
from typing import Tuple, List
import numpy as np
import pandas as pd
from numba import njit
import h5py

def pegar_variaveis_cgns(arquivo_h5py: h5py, variavel: str) -> np.array:
    """
    Essa função extrai valores da variável escolhida como input do .cgns,
    retorna como um array da variável em cada coordenada.
    Inputs: 
        Arquivo processado pelo h5py.File() e a variável sob forma de string.
    *Lembrando que, para ver os tipos de variável que existe, basta desco-
    mentar a linha no meio do código que printa k2.
    Outputs:
        Array contendo a variável já alinhada para caso seja feito um dataframe 
        contendo essa variável e cada coordenada.
    """

    # Criamos, primeiramente um array numpy vazio
    all_together: np.array = []
    search = 'Results3D/Level#01_Zone#0000000001'
    # Pegamos a zona a se olhar (sempre será essa)
    zone_to_look = arquivo_h5py[search]
    # Podemos varrer a área toda buscando pelo k2 certo, cada k2 é uma variável
    for k, v in zone_to_look.items():
        if k != ' data':
            for k2, v2 in arquivo_h5py[search + "/" + k].items():
                # DESCOMENTE ESSA LINHA ABAIXO PARA LISTAR AS VARIÁVEIS:
                # print(k2)
                if k2 == variavel: # Buscamos k2 igual a variável que queremos
                    data = np.array(arquivo_h5py[search + "/" + k + "/" + k2 + "/" + ' data'][:])[1:-1, 1:-1, 1:-1]
                    all_together.append(data)

    # Aqui, podemos fazer algumas transformações para deixar no mesmo formato das 
    # coordendas, podendo alinhar um com o outro.
    final_array = np.array(all_together)
    final_array = final_array[-3:, ...]
    final_array = np.mean(final_array, axis=0)
    final_array = final_array.reshape(-1)
   
    return final_array

@staticmethod
@njit
def _transformar_coordenadas(
    grid_x: np.array, grid_y: np.array, grid_z: np.array) -> Tuple[np.array, np.array, np.array]:

    """
    Essa função é interna. Não se preocupe com ela, você provavelmente não vai 
    precisar utilizar. A função pegar_coordenadas() a utiliza para transformar
    a estrutura que o h5py retorna por padrão na estrutura pronta para alinhar
    a uma estrutura mais comum.

    Inputs:
    Coordenadas dos pontos.

    Outputs:
    Coordenadas corrigidas.
    """

    # Coordenada X
    grid_x = grid_x[1:, 1:, :]
    x_grid2 = (grid_x[:, :, :-1] + grid_x[:, :, 1:]) / 2
    x_flat = np.ascontiguousarray(x_grid2).reshape(-1)

    # Coordenada Y
    y_grid2 = (grid_y[:, :-1, :] + grid_y[:, 1:, :]) / 2
    y_flat = np.ascontiguousarray(y_grid2[:-1, :, 1:]).reshape(-1)

    # Coordenada Z
    z_grid = (grid_z[:-1, :-1, 1:] + grid_z[1:, :-1, 1:]) / 2
    z_flat = np.ascontiguousarray(z_grid).reshape(-1)

    return x_flat, y_flat, z_flat

def pegar_coordenadas(cgns_file: h5py) -> Tuple[np.array, np.array, np.array]:

    """
    Essa função tem bastante importância. Ela consegue extrair as coordenadas
    do .cgns pelo h5py, porém essa biblioteca acaba entregando esses pontos
    em uma formatação meio esquisita. Então, o código também consegue fazer
    transformações nesses dados e entregá-los em um formato bacana.

    Inputs:
    Coordenadas dos pontos.

    Outputs:
    Coordenadas corrigidas.
    """

    # Tenta pegar as coordenadas, caso o arquivo não esteja íntegro, dá erro.
    try:
        grid_x = cgns_file['Results3D/Level#01_Zone#0000000001/GridCoordinates/ link/CoordinateX/ data']
        grid_y = cgns_file['Results3D/Level#01_Zone#0000000001/GridCoordinates/ link/CoordinateY/ data']
        grid_z = cgns_file['Results3D/Level#01_Zone#0000000001/GridCoordinates/ link/CoordinateZ/ data']

    except Exception as e:
        print(f"Erro ao ler coordenadas do arquivo: {cgns_file}")
        return
    
    # Transforma em array do numpy
    grid_x = np.array(grid_x)
    grid_y = np.array(grid_y)
    grid_z = np.array(grid_z)

    # Roda uma função interna para transformar as coordenadas
    x_grid, y_grid, z_grid = _transformar_coordenadas(grid_x, grid_y, grid_z)

    return x_grid, y_grid, z_grid

def gerar_df_cgns(cgns_path: str, variaveis: list) -> pd.DataFrame:

    """
    Função com o intuito de criar um DataFrame fácil de se manipular com as informações do .cgns lido. Para este df, é muito mais
    fácil atribuir um ID a cada paralelogramo do grid e também descobrir os valores máximos e mínimos das nossas coordenadas.

    Inputs: 
    - cgns_path (str): String com o path do cgns.

    Outputs: 
    - df_cgns (list): Dataframe com as informações de x, y e z de todos os paralelogramos além da informações de temperatura de
    cada ponto.
    """
    cgns_file = h5py.File(cgns_path, "r")

    # Pegando as coordenadas do cgns
    x, y, z = pegar_coordenadas(cgns_file)
    # Criando o dataframe e atribuindo os valores
    df_cgns = pd.DataFrame()
    df_cgns["X"] = x
    df_cgns["Y"] = y
    df_cgns["Z"] = z
    for var in variaveis:
        # Importando o array de cada variável da outra função
        var_array = pegar_variaveis_cgns(cgns_file, var)
        df_cgns[var] = var_array

    return df_cgns

def verificar_dominio(df_cgns: pd.DataFrame, df_pontos: pd.DataFrame) -> pd.DataFrame:
    
    """
    Essa função verifica quais dos pontos estão dentro do intervalo do .cgns e ao final entrega um Dataframe contendo uma nova 
    coluna com true ou false para se está no domínio ou não.
    
    Inputs:
    - df_cgns (pd.Dataframe): Entra com os dados do cgns para análise.
    - df_pontos (pd.Dataframe): Entra com os dados dos pontos para análise e posterior modificação.

    Outputs:
    - df_pontos_filtrado (pd.Dataframe): Retorno com os dados finais com uma nova coluna de booleanos que filtra os dados que se 
    encaixaram no domínio proposto.
    """

    # Pegamos os valores máximos e mínimos de cada coordenada 
    min_x, max_x = df_cgns["X"].min(), df_cgns["X"].max()
    min_y, max_y = df_cgns["Y"].min(), df_cgns["Y"].max()
    min_z, max_z = df_cgns["Z"].min(), df_cgns["Z"].max()

    # Verificamos se cada valor dentro do df está entre o máximo e o mínimo, portanto, estando dentro do domínio do df
    df_pontos["Dentro_dominio"] = ((df_pontos["X"].between(min_x, max_x)) & (df_pontos["Y"].between(min_y, max_y)) &
    (df_pontos["Z"].between(min_z, max_z)))
    df_pontos_filtrado = df_pontos

    return df_pontos_filtrado


def _encontrar_pos_prox_e_dist(df_cgns: pd.DataFrame, df_pontos: pd.DataFrame) -> pd.DataFrame:

    """
    Essa função é interna, será também utilizada para fazer a conta da interpolação. Sobre o que ela faz em si, é compli-
    cado de explicar, mas caso você veja espacialmente, verá que precisaremos pegar 4 paralelepípedos do .cgns, já que 
    pegamos os dois mais próximos em cada eixo para interpolar. Caso você pense que deveriam ser 6 basta perceber que 
    existe um paralelepípedo comum entre eles, o que o ponto em questão está contido. Para realizar a conta de interpolação 
    em cada eixo, precisamos descobrir aonde está este paralelepípedo no qual o ponto está contido para manter constante 
    suas coordenadas na conta enquanto interpolamos cada eixo. O objetivo dessa função é descobrir, para cada linha do df_pontos,
    o x_proximo, y_proximo e z_proximo, as coordenadas do paralelepípedo onde o ponto está contido e também o x_distante, 
    y_distante e z_distante, coordenadas que também serão úteis durante a interpolação.

    Input:
    - df_cgns (pd.DataFrame): Dataframe responsável por armazenar informações do .cgns.
    - df_pontos (pd.DataFrame): Dataframe responsável por armazenas as informações dos pontos.

    Output:
    - df_pontos_completo (pd.DataFrame): O dataframe contendo essa informação do paralelepípedo mais próximo e o segundo
    mais próximo do ponto analisado.
    """
    # Antes de tudo, aplicamos o filtro dos valores dentro do intervalo.
    df_pontos = verificar_dominio(df_cgns, df_pontos)

    # Primeiramente, para preservar a eficiência computacional e o bom senso, precisamos dropar as duplicatas e também organizar
    # os valores. Aqui, perceba que pegamos apenas os valores dentro do domínio, isso servirá futuramente para calcular a tempe-
    # ratura. Para valores fora do domínio a temperatura é fixa e considerada 20ºC.
    temps_x_cgns = list(df_cgns.copy()["X"].sort_values().drop_duplicates())
    temps_x_pontos = list(df_pontos.query('Dentro_dominio==True')["X"].sort_values())
    temps_y_cgns = list(df_cgns.copy()["Y"].sort_values().drop_duplicates())
    temps_y_pontos = list(df_pontos.query('Dentro_dominio==True')["Y"].sort_values())
    temps_z_cgns = list(df_cgns.copy()["Z"].sort_values().drop_duplicates())
    temps_z_pontos = list(df_pontos.query('Dentro_dominio==True')["Z"].sort_values())
    
    # Aqui, loopamos para cada variável - preferi deixar aberto desse jeito ao invés de colocar outro loop envolvendo os 3 eixos
    # exatamente para melhorar a legibilidade do que está sendo feito. A lógica em si é bem simples. Como os dois estão em ordem,
    # andamos com o índice até que ele se encontre em uma situação onde o da frente é maior e o de trás é menor. Pegamos o mais
    # próximo do valor da coordenada contida no df_pontos e chamamos ele de proximo, o outro é chamado de distante.
    # Isso é particularmente útil porque o ponto está contido no paralelepípedo em que todas as coordenadas são as mais próximas,
    # então, para achar a temperatura do paral. do lado, precisamos das coordenadas do que ele está contido.
    x_distante = []
    x_proximo = []
    # Varrendo as listas
    for x_ptos in range(len(temps_x_pontos)):
        for x_cgns in range(len(temps_x_cgns)-1):
            # Se o da frente é maior e o de trás é menor, podemos prosseguir.
            if (temps_x_cgns[x_cgns] <= temps_x_pontos[x_ptos]) and (temps_x_cgns[x_cgns+1] >= temps_x_pontos[x_ptos]):
                # Quem é mais próximo, o maior valor ou o menor? Verificamos isso e adicionamos os valores.
                if abs(temps_x_cgns[x_cgns] - temps_x_pontos[x_ptos]) >= abs(temps_x_cgns[x_cgns+1] - temps_x_pontos[x_ptos]):
                    x_distante.append(temps_x_cgns[x_cgns])
                    x_proximo.append(temps_x_cgns[x_cgns+1])
                elif abs(temps_x_cgns[x_cgns] - temps_x_pontos[x_ptos]) <= abs(temps_x_cgns[x_cgns+1] - temps_x_pontos[x_ptos]):
                    x_distante.append(temps_x_cgns[x_cgns+1])
                    x_proximo.append(temps_x_cgns[x_cgns])
                else:
                    print("Algo deu errado ao descobrir x_proximo e x_distante")
                break
    
    # São feitas as mesmas coisas para y. Lembrando, fiz desse jeito para melhorar a legibilidade do código
    y_distante = []
    y_proximo = []
    for y_ptos in range(len(temps_y_pontos)):
        for y_cgns in range(len(temps_y_cgns)-1):
            if (temps_y_cgns[y_cgns] <= temps_y_pontos[y_ptos]) and (temps_y_cgns[y_cgns+1] >= temps_y_pontos[y_ptos]):
                if abs(temps_y_cgns[y_cgns] - temps_y_pontos[y_ptos]) >= abs(temps_y_cgns[y_cgns+1] - temps_y_pontos[y_ptos]):
                    y_distante.append(temps_y_cgns[y_cgns])
                    y_proximo.append(temps_y_cgns[y_cgns+1])
                elif abs(temps_y_cgns[y_cgns] - temps_y_pontos[y_ptos]) <= abs(temps_y_cgns[y_cgns+1] - temps_y_pontos[y_ptos]):
                    y_distante.append(temps_y_cgns[y_cgns+1])
                    y_proximo.append(temps_y_cgns[y_cgns])
                else:
                    print("Algo deu errado ao descobrir y_proximo e y_distante")
                break

    # Idem para z
    z_distante = []
    z_proximo = []
    for z_ptos in range(len(temps_z_pontos)):
        for z_cgns in range(len(temps_z_cgns)-1):
            if (temps_z_cgns[z_cgns] <= temps_z_pontos[z_ptos]) and (temps_z_cgns[z_cgns+1] >= temps_z_pontos[z_ptos]):
                if abs(temps_z_cgns[z_cgns] - temps_z_pontos[z_ptos]) >= abs(temps_z_cgns[z_cgns+1] - temps_z_pontos[z_ptos]):
                    z_distante.append(temps_z_cgns[z_cgns])
                    z_proximo.append(temps_z_cgns[z_cgns+1])
                elif abs(temps_z_cgns[z_cgns] - temps_z_pontos[z_ptos]) <= abs(temps_z_cgns[z_cgns+1] - temps_z_pontos[z_ptos]):
                    z_distante.append(temps_z_cgns[z_cgns+1])
                    z_proximo.append(temps_z_cgns[z_cgns])
                else:
                    print("Algo deu errado ao descobrir z_proximo e z_distante")
                break

    
    # Por fim, temos que salvar no df_pontos final novamente. Vale lembrar que precisamos ordenar o df_pontos antes de adicionar para
    # adicionar no lugar certo, faz sentido?
    df_pontos = df_pontos.sort_values(by="X")
    df_pontos.loc[df_pontos["Dentro_dominio"] == True, "X_distante"] = x_distante
    df_pontos.loc[df_pontos["Dentro_dominio"] == True, "X_proximo"] = x_proximo
    df_pontos = df_pontos.sort_values(by="Y")
    df_pontos.loc[df_pontos["Dentro_dominio"] == True, "Y_distante"] = y_distante
    df_pontos.loc[df_pontos["Dentro_dominio"] == True, "Y_proximo"] = y_proximo
    df_pontos = df_pontos.sort_values(by="Z")
    df_pontos.loc[df_pontos["Dentro_dominio"] == True, "Z_distante"] = z_distante
    df_pontos.loc[df_pontos["Dentro_dominio"] == True, "Z_proximo"] = z_proximo
    df_pontos_completo = df_pontos.sort_values(by="ID")

    return df_pontos_completo

def interpolar_variaveis(df_cgns: pd.DataFrame, df_pontos: pd.DataFrame, variaveis: list) -> pd.DataFrame:
    """
    Interpola as temperaturas nos eixos X, Y e Z e faz uma média aritmética dos 3 resultados para produzir uma temperatura interpolada
    final. 

    Input:
    - df_cgns (pd.DataFrame): Dataframe responsável por armazenar informações do .cgns.
    - df_pontos (pd.DataFrame): Dataframe com todas as informações dos pontos.

    Output:
    - df_pontos_final (pd.DataFrame): O dataframe contendo o resultado final e objetivo do código, possui as variáveis interpoladas
    para cada um dos pontos inputados.
    """    
    # Antes de tudo, chamar a função anterior para pegar o df_completo sem variáveis
    df_pontos_completo =  _encontrar_pos_prox_e_dist(df_cgns, df_pontos)

    # AINDA NAO ESTAMOS AVALIANDO VALORES PADRÃO.
    # Para todas as coordenadas fora do domínio, é definido como temperatura 20ºC, em Kelvins: 293.15K
    # df_pontos_completo.loc[df_pontos_completo['Dentro_dominio'] == False, 'Temperatura'] = 293.15

    for var in variaveis:
        # Essa espécie de closure tem a função de interpolar as temperaturas para cada eixo recebendo as informações necessárias para tal. 
        def interpolate(var_distante, var_proximo, pos_distante, pos_proximo, pos_pto):
            # Verifica se o tamanho é maior que 0, ou seja, a lista faz sentido.
            if var_distante.size > 0 and var_proximo.size > 0:
                var_distante = var_distante[0]
                var_proximo = var_proximo[0]
                
                # Verifica quem é o maior entre o distante e o próximo, isso vai ser útil para se interpolar os termos.
                d_pos = pos_distante - pos_proximo
                if d_pos != 0:
                    # Se distante > proximo
                    if d_pos > 0:
                        proporcao = (pos_distante-pos_proximo) / (pos_pto-pos_proximo)
                        return ((var_distante-var_proximo) + proporcao*var_proximo) / proporcao
                    # Se proximo > distante
                    else:
                        proporcao = (pos_proximo-pos_distante) / (pos_pto-pos_distante)
                        return ((var_proximo-var_distante) + proporcao*var_distante) / proporcao
                else:
                    return var_proximo
            else:
                return np.nan

        # Criamos as listas que vão conter as temperaturas interpoladas para cada eixo
        interp_x, interp_y, interp_z = [], [], []

        # Loopamos todas as linhas do df
        for i in range(len(df_pontos_completo)):
            #  Pegamos as informações necessárias inicialmente
            x_proximo, y_proximo, z_proximo = df_pontos_completo.loc[i, ["X_proximo", "Y_proximo", "Z_proximo"]]
            x_distante, y_distante, z_distante = df_pontos_completo.loc[i, ["X_distante", "Y_distante", "Z_distante"]]
            x_pto, y_pto, z_pto = df_pontos_completo.loc[i, ["X", "Y", "Z"]]
            
            # Precisamos lidar com a busca no df_cgns da linha que representa equivalentemente o que queremos
            var_distante_x = df_cgns.loc[(df_cgns["X"] == x_distante) & (df_cgns["Y"] == y_proximo) & (df_cgns["Z"] == z_proximo), var].values
            var_proximo_x = df_cgns.loc[(df_cgns["X"] == x_proximo) & (df_cgns["Y"] == y_proximo) & (df_cgns["Z"] == z_proximo), var].values
            var_distante_y = df_cgns.loc[(df_cgns["X"] == x_proximo) & (df_cgns["Y"] == y_distante) & (df_cgns["Z"] == z_proximo), var].values
            var_proximo_y = df_cgns.loc[(df_cgns["X"] == x_proximo) & (df_cgns["Y"] == y_proximo) & (df_cgns["Z"] == z_proximo), var].values
            var_distante_z = df_cgns.loc[(df_cgns["X"] == x_proximo) & (df_cgns["Y"] == y_proximo) & (df_cgns["Z"] == z_distante), var].values
            var_proximo_z = df_cgns.loc[(df_cgns["X"] == x_proximo) & (df_cgns["Y"] == y_proximo) & (df_cgns["Z"] == z_proximo), var].values
            
            # Usamos a função para calcular a interpolação
            interp_x.append(interpolate(var_distante_x, var_proximo_x, x_distante, x_proximo, x_pto))
            interp_y.append(interpolate(var_distante_y, var_proximo_y, y_distante, y_proximo, y_pto))
            interp_z.append(interpolate(var_distante_z, var_proximo_z, z_distante, z_proximo, z_pto))

        # Passamos as informações das listas pra o DataFrame
        df_pontos_completo = df_pontos_completo
        df_pontos_completo['Interp_x'] = interp_x
        df_pontos_completo['Interp_y'] = interp_y
        df_pontos_completo['Interp_z'] = interp_z

        # Finalmente, é adicionada a nova coluna de temperatura no DataFrame final
        df_pontos_completo.loc[df_pontos_completo['Dentro_dominio'] == True, var] = \
            (df_pontos_completo['Interp_x'] + df_pontos_completo['Interp_y'] + df_pontos_completo['Interp_z'])/3

    # Dropamos todas as colunas indesejadas para restar apenas a temperatura e as coordenadas do .fem
    df_pontos_final = df_pontos_completo.drop(columns=['X_distante', 'X_proximo', 'Y_distante', 'Y_proximo', 'Z_distante', 'Z_proximo', 'Interp_x', 'Interp_y', 'Interp_z'])

    return df_pontos_final


def main():
    arquivo = h5py.File(r"U:\FLACS\Jet Fire\M01\Cen10\101010.cgns")
    variaveis = ["Temperature", "Visibility", "TotHeatFlux", "MassFractionCO2", "MassFractionOxygen"]
    df_cgns = gerar_df_cgns(arquivo, variaveis)
    df_cgns.to_excel("df_cgns_101010.xlsx")

# def main():
#     pasta_cgns = r"U:\FLACS\Jet Fire\M01\Cen10" 
#     variaveis = ["Temperature", "Visibility", "TotHeatFlux", "MassFractionCO2", "MassFractionOxygen"]
#     df_pontos = pd.read_excel("./df_pontos.xlsx")
#     for arquivo in os.listdir(pasta_cgns):
#         if arquivo.endswith(".cgns"):
#             arquivo_h5py = h5py.File(os.path.join(pasta_cgns, arquivo), "r")
#             df_cgns = gerar_df_cgns(arquivo_h5py, variaveis)
#             # df_cgns.to_excel("df_cgns.xlsx")
#             # df_pontos_filtrado = verificar_dominio(df_cgns, df_pontos)
#             df_pontos_completo = interpolar_variaveis(df_cgns, df_pontos, variaveis)
#             df_pontos_completo.to_excel(f"./df_pontos_completo_{arquivo[:7]}.xlsx")

if __name__=="__main__":
    main()

