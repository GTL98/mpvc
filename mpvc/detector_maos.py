# --- Importar as bibliotecas --- #
import cv2
import math
import numpy as np
import mediapipe as mp
from typing import Any, List, Tuple


class DetectorMaos:
    """Classe responsável pela detecção de mãos."""
    def __init__(
            self,
            modo: bool = False,
            max_maos: int = 2,
            deteccao_confianca: float = 0.5,
            rastreio_confianca: float = 0.5,
            cor_pontos: Tuple[int, int, int] = (0, 0, 255),
            cor_conexoes: Tuple[int, int, int] = (255, 255, 255),
    ):
        """
        Função responsável por inicializar a classe.

        :param modo: Modo de captura da imagem. Se True, a detecção e rastreio serão feitos a todo momento;
        fica muito travado. Se False, não são feitas a detecção e rastreio a todo momento; pode perder por
        alguns instantes as marcações, porém não trava.

        :param max_maos: Quantidade máxima de mãos para serem detectadas.

        :param deteccao_confianca: Percentual da taxa de detecção da mão. Se for menor do que este limite,
        a deteção não ocorre.

        :param rastreio_confianca: Percentual da taxa de detecção da mão. Se for menor do que este limite,
        a detecção não ocorre.

        :param cor_pontos: Cor dos pontos.

        :param cor_conexoes: Cor das conexões.
        """
        # --- Inicializar os módulos de detecção das mãos --- #
        self.maos_mp = mp.solutions.hands
        self.maos = self.maos_mp.Hands(
            modo,
            max_maos,
            1,
            deteccao_confianca,
            rastreio_confianca
        )

        # --- Função para desenhar os pontos nas mãos --- #
        self.desenho = mp.solutions.drawing_utils

        # --- Confifuração do desenho dos pontos --- #
        self.desenho_pontos = self.desenho.DrawingSpec(color=cor_pontos)

        # --- Configuração do desenho das conexões --- #
        self.desenho_conexoes = self.desenho.DrawingSpec(color=cor_conexoes)

    def encontrar_maos(self, imagem: np.ndarray, desenho: bool = True) -> np.ndarray:
        """
        Função responsável por detectar a(s) mão(s).
        :param imagem: Imagem capturada.
        :param desenho: Desenhar os pontos e as conexões na(s) mão(s).
        :return: Retorna a imagem com a detecção.
        """
        # --- Converter a imagem de BGR para RGB --- #
        imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

        # --- Passar a imagem convertida para o detector --- #
        self.resultado = self.maos.process(imagem_rgb)

        # --- Verificar se alguma mão foi detectada --- #
        if self.resultado.multi_hand_landmarks:
            for pontos in self.resultado.multi_hand_landmarks:
                if desenho:
                    # --- Desenhar os pontos nas mãos detectadas --- #
                    self.desenho.draw_landmarks(
                        imagem,
                        pontos,
                        self.maos_mp.HAND_CONNECTIONS,
                        self.desenho_pontos,
                        self.desenho_conexoes
                    )

        return imagem

    def encontrar_pontos(
            self,
            imagem: np.ndarray,
            mao_num: int = 0,
            desenho: bool = True,
            cor: Tuple[int, int, int] = (255, 0, 255),
            raio: int = 7,
            ponto_detectado: Any = 0
    ) -> List:
        """
        Função responsável por encontrar a posição dps pontos da(s) mão(s).
        :param imagem: Imagem capturada.
        :param mao_num: Número da mão detectada.
        :param desenho: Desenhar o ponto encontrado.
        :param cor: Tupla com a cor do ponto (RGB).
        :param raio: Raio do círculo do ponto.
        :param ponto_detectado: Ponto a ser detectado.
        :return: Lista com os pontos detectados.
        """
        # --- Lista com os pontos detectados --- #
        lista_pontos = []

        # --- Garantir que o ponto_detectado seja uma lista --- #
        if isinstance(ponto_detectado, int):
            pontos_alvo = [ponto_detectado]
        else:
            pontos_alvo = ponto_detectado

        # --- Obter as dimensões da imagem capturada --- #
        altura, largura, _ = imagem.shape

        # --- Verificar se alguma mão foi detectada --- #
        if self.resultado.multi_hand_landmarks:
            for num_mao, landmark_mao in enumerate(self.resultado.multi_hand_landmarks):
                # --- Lista com os pontos da mão detectada --- #
                lista_pontos_mao = []
                # --- Obter as informações dos pontos --- #
                for id_, ponto in enumerate(landmark_mao.landmark):
                    if id_ in pontos_alvo:
                        # --- Transformar a posição do ponto de proporção para pixel --- #
                        centro_x, centro_y = int(ponto.x * largura), int(ponto.y * altura)

                        # --- Adicionar os pontos da mão detectda na lista --- #
                        lista_pontos_mao.append([id_, centro_x, centro_y])

                        # --- Colocar um círculo em um ponto --- #
                        if desenho:
                            cv2.circle(
                                imagem,
                                (centro_x, centro_y),
                                raio,
                                cor,
                                cv2.FILLED
                            )
                lista_pontos.append(lista_pontos_mao)

        return lista_pontos

    def encontrar_distancia(
            self,
            ponto_1: int,
            ponto_2: int,
            imagem: np.ndarray,
            mao_num: int = 0,
            desenho_linha: bool = True,
            desenho_ponto: bool = True,
            cor_linha: Tuple[int, int, int] = (255, 0, 255),
            cor_ponto: Tuple[int, int, int] = (0, 255, 0),
            espessura: int = 2,
            raio: int = 7
    ) -> Tuple[float, np.ndarray, List[int]]:
        """
        Função responsável por calcular a distância entre dois pontos da mão.
        :param ponto_1: ID do primeiro ponto.
        :param ponto_2: ID do segundo ponto.
        :param imagem: Imagem capturada.
        :param mao_num: Índice da mão detectada.
        :param desenho_linha: Se True, desenha uma linha entre os pontos e o centro.
        :param desenho_ponto: Se True, desenha o ponto médio entre os pontos selecionados.
        :param cor_linha: Cor da linha.
        :param cor_ponto: Cor do ponto médio.
        :param espessura: Espessura da linha que conecta os pontos.
        :param raio: Raio do ponto central da linha.
        :return: Uma tupla contendo a distância calculada, a imagem e uma lista com as coordenadas.
        """
        distancia = 0
        info_pontos = []

        # --- Verificar se alguma mão foi detectada e se o índice da mão existe --- #
        if self.resultado.multi_hand_landmarks and len(self.resultado.multi_hand_landmarks) > mao_num:
            mao_alvo = self.resultado.multi_hand_landmarks[mao_num]
            altura, largura, _ = imagem.shape

            # --- Extrair as coordenadas em pixels dos dois pontos --- #
            x_1 = int(mao_alvo.landmark[ponto_1].x * largura)
            y_1 = int(mao_alvo.landmark[ponto_1].y * altura)

            x_2 = int(mao_alvo.landmark[ponto_2].x * largura)
            y_2 = int(mao_alvo.landmark[ponto_2].y * altura)

            # --- Encontrar o ponto central entre os pontos --- #
            centro_x, centro_y = (x_1 + x_2) // 2, (y_1 + y_2) // 2
            info_pontos = [x_1, y_1, x_1, y_2, centro_x, centro_y]

            # --- Calcular a distância euclidiana --- #
            distancia = math.hypot(x_2 - x_1, y_2 - y_1)

            # --- Desenhar a linha e o ponto central --- #
            if desenho_linha:
                cv2.line(imagem, (x_1, y_1), (x_2, y_2), cor_linha, espessura)
            if desenho_ponto:
                cv2.circle(imagem, (centro_x, centro_y), raio, cor_ponto, cv2.FILLED)

        return distancia, imagem, info_pontos


def main():
    # --- Capturar o vídeo pela webcam --- #
    cap = cv2.VideoCapture(0)

    # --- Instanciar a classe do detector --- #
    detector = DetectorMaos()

    # --- Realizar a captura --- #
    while True:
        # --- Obter a imagem --- #
        _, imagem = cap.read()

        # --- Inverter a imagem --- #
        imagem = cv2.flip(imagem, 1)

        # --- Realizar a detecção das mãos --- #
        imagem = detector.encontrar_maos(imagem)

        # --- Lista com os pontos --- #
        lista_pontos = detector.encontrar_pontos(imagem)

        # --- Distância entre os pontos --- #
        if len(lista_pontos) != 0 and len(lista_pontos[0]) > 0:
            distancia, imagem, info = detector.encontrar_distancia(4, 8, imagem, desenho_ponto=False)
            cv2.putText(
                imagem,
                f'Dist: {int(distancia)}',
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2
            )

        # --- Mostrar a imagem de captura --- #
        cv2.imshow('Captura', imagem)

        # --- Tempo de atualização da captura --- #
        cv2.waitKey(1)


if __name__ == '__main__':
    main()