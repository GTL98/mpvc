# --- Importar as bibliotecas --- #
import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple

class DetectorMalhaRosto:
    def __init__(
            self,
            modo: bool = False,
            max_rosto: int = 2,
            refinar_pontos: bool = False,
            deteccao_confianca: float = 0.5,
            rastreio_confianca: float = 0.5,
            cor_pontos: Tuple = (255, 255, 255),
            cor_conexoes: Tuple = (0, 255, 0)
    ):
        """
        Função responsável por inicializar a classe.

        :param modo: Modo de captura da imagem. Se True, a detecção e o rastrio serão feitos a todo momento;
        deixa muito travado. Se False, não são feitas a detecção e rastrio a todo momento; pode perder
        pos alguns instântes as marcações, porém não trava.

        :param max_rostos: Quantidade máxima de rostos para serem detectados.

        :param refinar_pontos: Refinar as coordenadas dos pontos da boca e dos olhos.

        :param deteccao_confianca: Percentual da taxa de detecção do rosto. Se for menor do que este limite,
        a detecção não ocorre.

        :param rastreio_confianca:  Percentual da taxa do rastreio do rosto. Se for menor do que este limite,
        o rastreio não ocorre.

        :param cor_pontos: Cor dos pontos.

        :param cor_conexoes: Cor das conexões.
        """

        # --- Inicializar a função de detecção da malha do rosto --- #
        self.malha_rosto_mp = mp.solutions.face_mesh
        self.malha_rosto = self.malha_rosto_mp.FaceMesh(
            modo,
            max_rosto,
            refinar_pontos,
            deteccao_confianca,
            rastreio_confianca
        )
        self.malha_rosto_conexoes = mp.solutions.face_mesh_connections

        # --- Função para desenhar a malha no rosto --- #
        self.desenho = mp.solutions.drawing_utils

        # --- Configuração do desenho dos pontos --- #
        self.desenho_pontos = self.desenho.DrawingSpec(
            color=cor_pontos,
            thickness=1,
            circle_radius=1
        )

        # --- Configuração do desenho das conexões --- #
        self.desenho_conexoes = self.desenho.DrawingSpec(
            color=cor_conexoes,
            thickness=1
        )

    def encontrar_malha_rosto(
            self,
            imagem: np.ndarray,
            desenho: bool = True
    )-> np.ndarray:
        """
        Função responsável por detectar o(s) rosto(s) e aplicar a malha.
        :param imagem: Imagem capturada.
        :param desenho: Desenhar a(s) malha(s) no(s) rosto(s).
        :return: Retorna a imagem com a malha facial.
        """
        # --- Converter a imagem de BGR para RGB --- #
        imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

        # --- Passar a imagem convertida para o detector --- #
        self.resultado = self.malha_rosto.process(imagem_rgb)

        # --- Verificar se ocorreu detecção do rosto --- #
        if self.resultado.multi_face_landmarks:
            # --- Obter os pontos da malha facial --- #
            for ponto in self.resultado.multi_face_landmarks:
                if desenho:
                    # --- Desenhar os pontos da malha --- #
                    self.desenho.draw_landmarks(
                        imagem,
                        ponto,
                        self.malha_rosto_conexoes.FACEMESH_CONTOURS,
                        self.desenho_pontos,
                        self.desenho_conexoes
                    )

        return imagem

    def encontrar_pontos(self, imagem: np.ndarray) -> List:
        """
        Função responsável por informar os pontos da malha facial.
        :param imagem: Imagem capturada.
        :return: Retorna a lista com os pontos da(s) malha(s) do(s) rosto(s) detectado(s).
        """
        # --- Lista com os pontos do rosto --- #
        rostos = []

        # --- Obter as dimensões da imagem --- #
        altura, largura, _ = imagem.shape

        # --- Verificar se ocorreu a detecção do rosto --- #
        if self.resultado.multi_face_landmarks:
            # --- Obter os pontos da malha facial --- #
            for pontos in self.resultado.multi_face_landmarks:
                # --- Lista com a malha do rosto detectado --- #
                rosto = []

                # --- Obter a informação de cada ponto da malha --- #
                for  id_, ponto in enumerate(pontos.landmark):
                    # --- Converter os valores de proporção para pixels --- #
                    x, y = int(ponto.x * largura), int(ponto.y * altura)

                    # --- Adicionar à lista de rosto o ponto do rosto --- #
                    rosto.append([id_, x, y])

                # --- Adicionar a lista com os pontos do rosto à lista de rostos detectados --- #
                rostos.append(rosto)

        return rostos


def main():
    # --- Captura da webcam --- #
    cap = cv2.VideoCapture(0)

    # --- Instanciar o detector da malha facial --- #
    detector = DetectorMalhaRosto()

    # --- Loop de captura --- #
    while True:
        # --- Captura da imagem --- #
        _, imagem = cap.read()

        # --- Inverter a imagem --- #
        imagem = cv2.flip(imagem, 1)

        # --- Detectar a malha do rosto --- #
        imagem = detector.encontrar_malha_rosto(imagem)
        rostos = detector.encontrar_pontos(imagem)

        # --- Mostrar a imagem --- #
        cv2.imshow('Captura', imagem)

        # --- Tempo de atualização --- #
        cv2.waitKey(1)


if __name__ == '__main__':
    main()