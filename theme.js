(() => {
  const root = document.documentElement;
  const button = document.querySelector('.theme-toggle');
  if (!button) return;
  function reflect() {
    button.textContent = root.dataset.theme === 'dark' ? 'Light mode' : 'Dark mode';
  }
  button.hidden = false;
  reflect();
  button.addEventListener('click', () => {
    const next = root.dataset.theme === 'dark' ? 'light' : 'dark';
    root.dataset.theme = next;
    reflect();
    try { localStorage.setItem('cn-theme', next); } catch (_) {}
  });
})();
