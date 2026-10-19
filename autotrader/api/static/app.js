// Confirmation for dangerous forms: <form data-confirm="Really?">
document.addEventListener("submit", (e) => {
  const msg = e.target.dataset && e.target.dataset.confirm;
  if (msg && !window.confirm(msg)) e.preventDefault();
});
