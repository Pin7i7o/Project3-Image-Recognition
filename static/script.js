function previewImage(event) {
    const fileInput = event.target;
    const file = fileInput.files[0];
    const imagePreview = document.getElementById('imagePreview');

    // Check if file is an image
    if (file.type.match('image.*')) {
        const reader = new FileReader();
        reader.onload = function() {
            const img = document.createElement('img');
            img.src = reader.result;
            img.className = 'img-fluid'
            img.style.height = '100%';
            imagePreview.innerHTML = '';
            imagePreview.appendChild(img);
        }
        reader.readAsDataURL(file);
    } else {
        imagePreview.innerHTML = '<p>Selected file is not an image.</p>';
    }
}