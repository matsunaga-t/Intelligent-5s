from dotenv import load_dotenv
import os
from PIL import Image, ImageDraw
import sys
from matplotlib import pyplot as plt
from azure.core.exceptions import HttpResponseError
import requests
from collections import defaultdict
from enum import StrEnum

# import namespaces
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures, ImageAnalysisResult
from azure.core.credentials import AzureKeyCredential

class TermColor(StrEnum):
    Black = "\033[30m"
    Red = "\033[31m"
    Green = "\033[32m"
    Yellow = "\033[33m"
    Blue = "\033[34m"
    Magenta = "\033[35m"
    Cyan = "\033[36m"
    White = "\033[37m"
    bBlack = "\033[90m"
    bRed = "\033[91m"
    bGreen = "\033[92m"
    bYellow = "\033[93m"
    bBlue = "\033[94m"
    bMagenta = "\033[95m"
    bCyan = "\033[96m"
    bWhite = "\033[97m"
    reset = "\033[m"

_DEBUG = 0

def debug(s:str, color: TermColor):
    if _DEBUG:
        print(f"{color}{s}\033[m", end="")

def debugln(s:str, color: TermColor):
    if _DEBUG:
        print(f"{color}{s}\033[m")

class UnionSegTree:
    """
    Árvore de segmentos que armazena a união dos intervalos inseridos nela.
    """
    
    def __init__(self, segmentEndpoints: list[int]) -> None:
        """
        Inicializa a árvore de segmentos.
        
        Args
        -------------
            segmentEndpoints (list)
                lista com as extremidades dos segmentos.
        """
        
        # lista das extremidades dos segmentos, ordenado e único
        self._endpoints = sorted(segmentEndpoints)
        i = 0
        while i < len(self._endpoints) - 1:
            if self._endpoints[i] == self._endpoints[i + 1]:
                self._endpoints.pop(i + 1)
            else:
                i += 1
        
        # quantidade de elementos em _endpoints
        elementCount = len(self._endpoints)
        
        # obtido por contas
        nodeCount = 4 * elementCount - 3
        leafCount = 2 * elementCount - 1
        self._leafThreshold = nodeCount - leafCount
        
        # comprimento total armazenado por cada subárvore (nó ou intervalo)
        self._totalLength = [0] * nodeCount
        
        # número de segmentos que contém o intervalo indicado por cada nó
        self._count:list[int] = [0] * nodeCount
        
        # limites superiores e inferiores de cada nó
        self._maximumNodeValue = [self._endpoints[0]]  * nodeCount
        self._minimumNodeValue = [self._endpoints[-1]] * nodeCount
        
        power = 1
        while leafCount > power:
            power *= 2
        power //= 2
        offset = (leafCount - power) * 2
        
        # ajustando o valor das folhas
        leafIndex = self._leafThreshold
        i = 0
        while i < leafCount:
            self._maximumNodeValue[leafIndex + i] = self._maximumLeafValue( (i + offset) % leafCount)
            self._minimumNodeValue[leafIndex + i] = self._minimumLeafValue( (i + offset) % leafCount)
            i += 1
        
        i = nodeCount - 1
        while i > 0:
            if self._maximumNodeValue[(i - 1) // 2] < self._maximumNodeValue[i]:
                self._maximumNodeValue[(i - 1) // 2] = self._maximumNodeValue[i]
            if self._minimumNodeValue[(i - 1) // 2] > self._minimumNodeValue[i]:
                self._minimumNodeValue[(i - 1) // 2] = self._minimumNodeValue[i]
            i -= 1
    
    def insert(self, start: int, end: int) -> int:
        """
        Insere o intervalo fechado [`start`, `end`].
        
        Args
        ----------------
            start (int): O começo (mínimo) do intervalo.
            end (int): O fim (máximo) do intervalo.
        
        Returns
        ----------------
            val (int): O comprimento da união de todos os intervalos armazenados.
        """
        self._check(start, end)
        return self._update(start, end, 0, 1)
    
    def remove(self, start: int, end: int) -> int:
        """
        Remove o intervalo fechado [`start`, `end`].
        
        Args
        ----------------
            start (int): O começo (mínimo) do intervalo.
            end (int): O fim (máximo) do intervalo.
        
        Returns
        ----------------
            val (int): O comprimento da união de todos os intervalos armazenados.
        """
        self._check(start, end)
        return self._update(start, end, 0, -1)
    
    def length(self) -> int:
        """
        Retorna o comprimento da união dos intervalos armazenados.
        
        Returns
        ----------------
            val (int): O comprimento da união dos intervalos armazenados.
        """
        return self._totalLength[0]
    
    def _maximumLeafValue(self, index: int) -> int:
        """
        Calcula o máximo do intervalo de uma folha com base no seu índice.
        
        Função necessária pois apenas os valores das extremidades (conjunto unitário) são fornecidos.
        
        Args
        ----------------
            index (int): O índice da folha.
        
        Returns
        ----------------
            val (int): o maior valor contido na folha (máximo dela).
        """
        return self._endpoints[(index + 1) // 2] - (index % 2) + 1
    
    def _minimumLeafValue(self, index: int) -> int:
        """
        Calcula o mínimo do intervalo de uma folha com base no seu índice.
        
        Função necessária pois apenas os valores das extremidades (conjunto unitário) são fornecidos.
        
        Args
        ----------------
            index (int): O índice da folha.
        
        Returns
        ----------------
            val (int): o menor valor contido na folha (mínimo dela).
        """
        return self._endpoints[index // 2] + (index % 2)
    
    def _update(self, lowerBound: int, upperBound: int, node: int, operation: int) -> int:
        """
        Função interna que recalcula o comprimento da união dos intervalos armazenados na subárvore
        com a raiz no nó `node` após inserir o intervalo [`lowerBound`, `upperBound`].
        
        Apenas uma abstração para uso interno. O nó deve ser fornecido pois a classe usa uma árvore
        implícita, porque é mais fácil fazer assim no Python, mas, pelo menos, fica mais otimizada.
        
        Args
        ----------------
            lowerBound (int): Limite inferior (mínimo) do intervalo que se quer inserir ou remover.
            upperBound (int): Limite superior (máximo) do intervalo que se quer inserir ou remover.
            node (int): O nó (raiz) atual.
            operation (int): `+1` para inserir um intervalo, `-1` para removê-lo.
        
        Returns
        ----------------
            val (int): Novo comprimento armazenado na subárvore.
            
        """
        nodeMaximum = self._maximumNodeValue[node]
        nodeMinimum = self._minimumNodeValue[node]
        
        leftChild = node * 2 + 1
        rightChild = node * 2 + 2
        
        power = 1
        while node >= (1 << power) - 1:
            power += 1
        debug(f"{"   " * power}{node} -> ", TermColor.bBlue)
        
        if lowerBound <= nodeMinimum and nodeMaximum <= upperBound:
            
            debugln(f"{nodeMinimum}; {nodeMaximum}", TermColor.Green)
            
            self._count[node] += operation
            if self._count[node] > 0:
                if node >= self._leafThreshold:
                    self._totalLength[node] = 1
                else:
                    self._totalLength[node] = nodeMaximum - nodeMinimum
            else:
                self._count[node] = 0
                if node >= self._leafThreshold:
                    self._totalLength[node] = 0
                else:
                    self._totalLength[node] = self._totalLength[leftChild] + self._totalLength[rightChild] 
            return self._totalLength[node]
        
        debugln(f"{nodeMinimum}; {nodeMaximum}", TermColor.Yellow)
        
        if node >= self._leafThreshold:
            return self._totalLength[node]
        
        totalChildLength = 0
        
        if lowerBound < self._maximumNodeValue[leftChild]:
            totalChildLength += self._update(lowerBound, upperBound, leftChild, operation)
        if upperBound >= self._minimumNodeValue[rightChild]:
            totalChildLength += self._update(lowerBound, upperBound, rightChild, operation)
        
        if self._count[node] == 0:
            self._totalLength[node] = totalChildLength
        return self._totalLength[node]
    
    def _check(self, start: int, end: int) -> None:
        """
        Confere se os valores inseridos são válidos. Eles precisam ser `int` e estarem contidos na lista das extremidades.
        
        Args
        ----------------
            start (int): O começo (mínimo) do intervalo.
            end (int): O fim (máximo) do intervalo.
        """
        if not ( isinstance(start, int) and isinstance(end, int)):
            raise TypeError("`start` and `end` must be int")
        if start not in self._endpoints or end not in self._endpoints:
            raise ValueError(f"`start` and `end` must be one of {self._endpoints}")
    

def show_objects(image_filename, detected_objects):
    print ("\nAnnotating objects...")

    # Prepare image for drawing
    image = Image.open(image_filename)
    fig = plt.figure(figsize=(image.width/100, image.height/100))
    plt.axis('off')
    draw = ImageDraw.Draw(image)
    color = 'cyan'

    for detected_object in detected_objects:
        # Draw object bounding box
        r = detected_object.bounding_box
        bounding_box = ((r.x, r.y), (r.x + r.width, r.y + r.height)) 
        draw.rectangle(bounding_box, outline=color, width=3)
        plt.annotate(detected_object.tags[0].name,(r.x, r.y), backgroundcolor=color)

    # Save annotated image
    plt.imshow(image)
    plt.tight_layout(pad=0)
    objectfile = 'objects.jpg'
    fig.savefig(objectfile)
    print('  Results saved in', objectfile)

def show_people(image_filename, detected_people):
    print ("\nAnnotating objects...")

    # Prepare image for drawing
    image = Image.open(image_filename)
    fig = plt.figure(figsize=(image.width/100, image.height/100))
    plt.axis('off')
    draw = ImageDraw.Draw(image)
    color = 'cyan'

    for detected_person in detected_people:
        if detected_person.confidence > 0.0:
            # Draw object bounding box
            r = detected_person.bounding_box
            bounding_box = ((r.x, r.y), (r.x + r.width, r.y + r.height))
            draw.rectangle(bounding_box, outline=color, width=3)

    # Save annotated image
    plt.imshow(image)
    plt.tight_layout(pad=0)
    peoplefile = 'people.jpg'
    fig.savefig(peoplefile)
    print('  Results saved in', peoplefile)

def getOccupiedArea(result: ImageAnalysisResult, tags: set[str] = set(), minimumConfidence: float = .2) -> float:
    if result.objects is None or len(tags) == 0:
        return 0.0
    
    yValues: list[int] = []
    startingPoints = defaultdict(list)
    endingPoints = defaultdict(list)
    
    # seleção dos objetos
    for object in result.objects.list:
        if object.tags[0].name in tags and object.tags[0].confidence >= minimumConfidence:
            yValues.append(object.bounding_box.y)
            yValues.append(object.bounding_box.y + object.bounding_box.height)
            
            startingPoints[object.bounding_box.x].append(object.bounding_box.y)
            startingPoints[object.bounding_box.x].append(object.bounding_box.y + object.bounding_box.height)
            
            endingPoints[object.bounding_box.x + object.bounding_box.width].append(object.bounding_box.y)
            endingPoints[object.bounding_box.x + object.bounding_box.width].append(object.bounding_box.y + object.bounding_box.height)

    xValues = sorted(set(startingPoints) | set(endingPoints))
    
    oldX = 0
    yLength = 0
    
    totalArea = 0
    intervals = UnionSegTree(yValues)

    #debug
    debugln(f"x = {xValues}", TermColor.Red)
    debugln(f"y = {intervals._endpoints}", TermColor.Red)

    for x in xValues:
        totalArea += (x - oldX) * yLength
        
        #debug
        debugln(f"x= {x:3d}  y= {yLength:3d}  t= {totalArea}", TermColor.Red)
        
        for i in range(0, len(startingPoints[x]), 2):
            #debug
            debugln(f"start = {startingPoints[x][i]} -> {startingPoints[x][i + 1]}", TermColor.Red)
            
            intervals.insert(startingPoints[x][i], startingPoints[x][i + 1])
            
        for i in range(0, len(endingPoints[x]), 2):
            #debug
            debugln(f"end = {endingPoints[x][i]} -> {endingPoints[x][i + 1]}", TermColor.Red)
            
            intervals.remove(endingPoints[x][i], endingPoints[x][i + 1])
        
        oldX = x
        yLength = intervals.length()
    
    return totalArea