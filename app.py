from flask import Flask, render_template, request, redirect, url_for, jsonify, session, send_file
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import os
import base64
from io import BytesIO
import threading
from flask import Response




app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Changez cette clé en production


# Liste des classes
class_labels = ['Engagement', 'Frustration', 'Confusion', 'Boredom']


def load_selected_model(selected_model):
    if selected_model == 'CNN':
        print("Loading CNN model.")  # Debug
        return load_model('cnn.h5')  # Assurez-vous que le fichier cnn.h5 est présent
    elif selected_model == 'LSTM-CNN':
        print("Loading LSTM-CNN model.")  # Debug
        return load_model('model_lstm_cnn.h5')  # Assurez-vous que le fichier model_lstm_cnn.h5 est présent
    print("No valid model selected.")  # Debug
    return None

@app.route('/select_model', methods=['POST'])
def select_model():
    model = request.form.get('model')
    if model:
        session['selected_model'] = model
        return jsonify({'message': f'Modèle {model} sélectionné avec succès.'}), 200
    else:
        return jsonify({'error': 'Aucun modèle sélectionné.'}), 400

def preprocess_image(image):
    image = image.resize((224, 224))  # Redimensionner l'image
    image = np.array(image) / 255.0  # Normalisation
    image = np.expand_dims(image, axis=0)  # Ajouter une dimension pour le batch
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify_images', methods=['GET', 'POST'])
def classify_images():
    if request.method == 'POST':
        selected_model = request.form.get('model')
        if not selected_model:
            print("Erreur : Aucun modèle sélectionné.")
            return jsonify({'error': 'Veuillez sélectionner un modèle.'}), 400
        # Stocker le modèle sélectionné dans la session
        session['selected_model'] = selected_model
        print(f"Modèle sélectionné : {selected_model}")  # Debug
        return jsonify({'message': 'Modèle sélectionné avec succès.'}), 200
    return render_template('classify_images.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    selected_model = session.get('selected_model')
    if not selected_model:
        print("Erreur : Aucun modèle sélectionné lors de l'upload de l'image.")
        return jsonify({'error': 'Aucun modèle sélectionné.'}), 400

    if 'image' not in request.files:
        print("Erreur : Aucune image téléchargée.")
        return jsonify({'error': 'Aucune image téléchargée.'}), 400

    image_file = request.files['image']

    if image_file.filename == '':
        print("Erreur : Nom de fichier vide.")
        return jsonify({'error': 'Nom de fichier vide.'}), 400

    try:
        print("Sauvegarde de l'image uploadée...")
        # Sauvegarder l'image temporairement
        image_path = os.path.join('static', 'temp_image.jpg')
        image_file.save(image_path)
        print(f"Image sauvegardée à {image_path}")

        print("Ouverture du fichier image...")
        # Charger et prétraiter l'image
        image = Image.open(image_path).convert('RGB')  # Assurez-vous que l'image est en RGB
        processed_image = preprocess_image(image)  # Fonction de prétraitement

        print("Chargement du modèle sélectionné...")
        # Charger le modèle sélectionné
        model = load_selected_model(selected_model)
        if model is None:
            print("Erreur : Modèle non chargé.")
            return jsonify({'error': 'Modèle non chargé.'}), 500
        print("Modèle chargé avec succès.")

        # Faire une prédiction
        predictions = model.predict(processed_image)
        if predictions is None or len(predictions) == 0:
            print("Erreur : Prédiction invalide.")
            return jsonify({'error': 'Prédiction invalide.'}), 500

        predicted_class_index = np.argmax(predictions[0])
        predicted_class_label = class_labels[predicted_class_index]
        predicted_class_probability = predictions[0][predicted_class_index]

        print(f"Prédiction : {predicted_class_label} ({predicted_class_probability:.2f})")

        # Superposer le texte de la prédiction sur l'image
        label = f"{predicted_class_label}: {predicted_class_probability:.2f}"
        image_cv = np.array(image)
        cv2.putText(image_cv, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Sauvegarder l'image annotée
        annotated_image_path = os.path.join('static', 'annotated_image.jpg')
        cv2.imwrite(annotated_image_path, image_cv)
        print(f"Image annotée sauvegardée à {annotated_image_path}")

        # Supprimer l'image temporaire
        os.remove(image_path)
        print("Image temporaire supprimée.")

        # Retourner le lien de téléchargement de l'image annotée
        download_link = url_for('static', filename='annotated_image.jpg')
        print(f"Image traitée disponible pour téléchargement à {download_link}")
        return jsonify({'download_link': download_link}), 200

    except Exception as e:
        print(f"Erreur lors de la classification des images : {e}")
        return jsonify({'error': 'La classification de l\'image a échoué.'}), 500


### Route pour la Sélection du Modèle Vidéo ###
@app.route('/classify_video', methods=['GET', 'POST'])
def classify_video():
   if request.method == 'POST':
       selected_model = request.form.get('model')
       if not selected_model:
           print("Erreur : Aucun modèle sélectionné.")
           return jsonify({'error': 'Veuillez sélectionner un modèle.'}), 400
       # Stocker le modèle sélectionné dans la session
       session['selected_model'] = selected_model
       print(f"Modèle sélectionné : {selected_model}")  # Debug
       return jsonify({'message': 'Modèle sélectionné avec succès.'}), 200
   return render_template('classify_video.html')


### Route pour le Téléchargement de la Vidéo ###
@app.route('/upload_video', methods=['POST'])
def upload_video():
   selected_model = session.get('selected_model')
   if not selected_model:
       print("Erreur : Aucun modèle sélectionné lors de l'upload de la vidéo.")
       return jsonify({'error': 'Aucun modèle sélectionné.'}), 400


   if 'video' not in request.files:
       print("Erreur : Aucune vidéo téléchargée.")
       return jsonify({'error': 'Aucune vidéo téléchargée.'}), 400


   video = request.files['video']


   if video.filename == '':
       print("Erreur : Nom de fichier vide.")
       return jsonify({'error': 'Nom de fichier vide.'}), 400


   if video:
       try:
           print("Sauvegarde de la vidéo uploadée...")
           # Sauvegarder la vidéo temporairement
           video_path = os.path.join('static', 'temp_video.mp4')
           video.save(video_path)
           print(f"Vidéo sauvegardée à {video_path}")


           print("Ouverture du fichier vidéo...")
           # Traiter la vidéo
           cap = cv2.VideoCapture(video_path)


           if not cap.isOpened():
               print("Erreur : Impossible d'ouvrir le fichier vidéo.")
               return jsonify({'error': 'Impossible d\'ouvrir la vidéo.'}), 500


           # Récupérer les propriétés de la vidéo
           width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
           height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
           fps = int(cap.get(cv2.CAP_PROP_FPS))
           codec = cv2.VideoWriter_fourcc(*'XVID')


           # Définir le chemin de la vidéo de sortie
           output_video_path = os.path.join('static', 'output_video.avi')
           out = cv2.VideoWriter(output_video_path, codec, fps, (width, height))
           print(f"Vidéo de sortie définie à {output_video_path}")


           print("Chargement du modèle sélectionné...")
           # Charger le modèle sélectionné
           model = load_selected_model(selected_model)
           if model is None:
               print("Erreur : Modèle non chargé.")
               return jsonify({'error': 'Modèle non chargé.'}), 500
           print("Modèle chargé avec succès.")


           frame_count = 0
           print("Début du traitement des frames...")
           # Traiter chaque frame
           while cap.isOpened():
               ret, frame = cap.read()
               if not ret:
                   break


               frame_count += 1
               if frame_count % 30 == 0:
                   print(f"Traitement de la frame {frame_count}")


               # Redimensionner et prétraiter la frame
               resized_frame = cv2.resize(frame, (224, 224))
               processed_frame = np.array(resized_frame) / 255.0
               processed_frame = np.expand_dims(processed_frame, axis=0)


               # Faire une prédiction
               predictions = model.predict(processed_frame)
               if predictions is None or len(predictions) == 0:
                   print("Erreur : Prédiction invalide.")
                   break


               predicted_class_index = np.argmax(predictions[0])
               predicted_class_label = class_labels[predicted_class_index]
               predicted_class_probability = predictions[0][predicted_class_index]


               # Ajouter le label à la frame
               label = f"{predicted_class_label}: {predicted_class_probability:.2f}"
               cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


               # Écrire la frame modifiée dans la vidéo de sortie
               out.write(frame)


           print("Libération des ressources...")
           # Libérer les ressources
           cap.release()
           out.release()
           print("Ressources libérées.")


           print("Suppression de la vidéo temporaire...")
           # Optionnel : Supprimer la vidéo temporaire
           os.remove(video_path)
           print("Vidéo temporaire supprimée.")


           # Retourner le lien de téléchargement de la vidéo traitée
           download_link = url_for('static', filename='output_video.avi')
           print(f"Vidéo traitée disponible pour téléchargement à {download_link}")
           return jsonify({'download_link': download_link}), 200


       except Exception as e:
           print(f"Erreur lors de la classification des vidéos : {e}")
           return jsonify({'error': 'La classification de la vidéo a échoué.'}), 500


   return jsonify({'error': 'Aucune vidéo téléchargée.'}), 400


# Route pour la prédiction en temps réel
@app.route('/real-time', methods=['GET'])
def real_time_prediction():
    selected_model = session.get('selected_model')
    if not selected_model:
        return redirect(url_for('classify_video'))  # Redirige vers la sélection du modèle si non sélectionné
    return render_template('real_time.html')

# Génération des frames pour le flux vidéo
def generate_frames(model):
    global stop_thread
    cap = cv2.VideoCapture(0)  # Ouvre la webcam
    if not cap.isOpened():
        print("Erreur : Webcam non accessible.")
        return

    while not stop_thread:
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (224, 224))
        processed_frame = np.array(resized_frame) / 255.0
        processed_frame = np.expand_dims(processed_frame, axis=0)

        predictions = model.predict(processed_frame)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class_label = class_labels[predicted_class_index]
        predicted_class_probability = predictions[0][predicted_class_index]

        label = f"{predicted_class_label}: {predicted_class_probability:.2f}"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    selected_model = session.get('selected_model')
    if not selected_model:
        return jsonify({'error': 'Aucun modèle sélectionné.'}), 400

    model = load_selected_model(selected_model)
    if model is None:
        return jsonify({'error': 'Modèle non chargé.'}), 500

    return Response(generate_frames(model),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_real_time', methods=['POST'])
def start_real_time():
    global stop_thread
    data = request.get_json()
    selected_model = data.get('model')
    if not selected_model:
        return jsonify({'error': 'Aucun modèle sélectionné.'}), 400

    # Stocker le modèle sélectionné dans la session
    session['selected_model'] = selected_model
    stop_thread = False
    return jsonify({'message': 'Prédiction en temps réel démarrée.'}), 200

@app.route('/stop_real_time', methods=['POST'])
def stop_real_time():
    global stop_thread
    stop_thread = True
    return jsonify({'message': 'Prédiction en temps réel arrêtée.'}), 200


if __name__ == '__main__':
   app.run(debug=True)



