self.addEventListener('fetch', function(event) {
    // Minimal service worker to allow Home Screen installation
    event.respondWith(fetch(event.request));
});