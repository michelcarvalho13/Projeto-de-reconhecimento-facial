import face_recognition
from PIL import Image, ImageDraw
import numpy as np

# banco de imagens
imagem001 = face_recognition.load_image_file("pessoas-conhecidas/alan1.jpg")
imagem001_encoding = face_recognition.face_encodings(imagem001)[0]

imagem002 = face_recognition.load_image_file('pessoas-conhecidas/Ronaldo que não é o Cristiano.jpeg')
imagem002_encoding = face_recognition.face_encodings(imagem002)[0]

imagem003 = face_recognition.load_image_file('pessoas-conhecidas/Ruan.jpeg')
imagem003_encoding = face_recognition.face_encodings(imagem003)[0]

imagem004 = face_recognition.load_image_file('pessoas-conhecidas/Michel.jpeg')
imagem004_encoding = face_recognition.face_encodings(imagem004)[0]

imagem005 = face_recognition.load_image_file('pessoas-conhecidas/victória.jpeg')
imagem005_encoding = face_recognition.face_encodings(imagem005)[0]

imagem006 = face_recognition.load_image_file('pessoas-conhecidas/Giordano.jpeg')
imagem006_encoding = face_recognition.face_encodings(imagem006)[0]


# bloco de imagens previamente armazenadas
# cada imagem adicionada precisa ser vinculada a versão encoding e separado por virgula
known_face_encodings = [
    imagem001_encoding,
    imagem002_encoding,
    imagem003_encoding,
    imagem004_encoding,
    imagem005_encoding,
    imagem006_encoding
]
# bloco de nomes previamente armazenados
# cada nome adicionado precisa ser como string e separado por virgula
known_face_names = [
    "Alan",
    "Ronaldo",
    "Ruan",
    "Michel",
    "Victória",
    "Giordano"
]

# carregamento de uma imagem para identificar os rostos
unknown_image = face_recognition.load_image_file("pessoas-n-conhecidas/imagem001.jpeg")

# análise da imagem para identificação dos rostos presentes
face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

# usa a biblioteca pillow para gerar um recorte do rosto e testar com alguma do banco de dados
pil_image = Image.fromarray(unknown_image)
draw = ImageDraw.Draw(pil_image)

# Loop para encontrar rostos desconhecidos
# (face_encoding e top, right, bottom, left estão definidas no API, logo,
# devem ter obrigatoriamente estes nomes para serem utilizadas)
# o loop a cima vai procurar um nome associado a database_rostos, se não encontrar, vai usar o nome 'indigente'
# sim, o 'else' das funções for.. else pode ser omitido neste tipo de estrutura, já subentende
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name = "Unknown"

    # esse aqui é o bloco do algoritmo que encaixa os nomes
    # face distances verifica na verdade a verossimilhança entre as imagens carregada no face_encoding e no
    # banco de imagens
    # usa a numpy.argmin para verificar a distância entre os pontos de rosto, quanto mais próximo do encontrado
    # no registro, melhor
    # quando o teste confere positivo, é um match e ele puxa do banco de nomes, o respectivo nome
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = known_face_names[best_match_index]

    # aqui é criado um retangulo da imagem (render), segue as proporções do vetor gerado pelo face_encoding (R4)
    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
    # cria uma legenda na imagem
    text_width, text_height = draw.textsize(name)
    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
    draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

# uma vez gerado o quadro de legenda, as informações são apagadas
del draw

pil_image.show()
