<template>
  <div class="container">
    <h1>Clasificador Gatos vs Perros</h1>

    <input type="file" accept="image/*" @change="onFileSelected" />

    <!-- Image preview -->
    <div v-if="preview" class="preview">
      <img :src="preview" alt="preview" />
    </div>

    <!-- Predict button -->
    <button :disabled="!selectedFile || loading" @click="predict">
      {{ loading ? "Procesando..." : "Predecir Imagen" }}
    </button>

    <!-- Prediction result -->
    <div v-if="result" class="result">
      <h2>Resultado:</h2>
      <p><strong>{{ result }}</strong></p>
    </div>

    <!-- Error -->
    <div v-if="error" class="error">
      {{ error }}
    </div>
  </div>
</template>

<script setup>
import { ref } from "vue";

// Load balancer public IP
const API_URL = "http://3.87.113.157/predict_image";

const selectedFile = ref(null);
const preview = ref(null);
const result = ref("");
const error = ref("");
const loading = ref(false);

function onFileSelected(e) {
  selectedFile.value = e.target.files[0];
  preview.value = URL.createObjectURL(selectedFile.value);
  result.value = "";
  error.value = "";
}

async function predict() {
  if (!selectedFile.value) return;

  loading.value = true;
  result.value = "";
  error.value = "";

  const formData = new FormData();
  formData.append("file", selectedFile.value);

  try {
    const res = await fetch(API_URL, {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    result.value = data.prediction;
  } catch (err) {
    error.value = "No se pudo conectar con el backend.";
  }

  loading.value = false;
}
</script>

<style>
.container {
  max-width: 450px;
  margin: auto;
  text-align: center;
  padding: 20px;
}

.preview img {
  width: 100%;
  margin-top: 10px;
  border-radius: 12px;
}

button {
  padding: 10px 16px;
  margin-top: 10px;
  cursor: pointer;
}

.error {
  color: red;
  margin-top: 15px;
}
</style>
