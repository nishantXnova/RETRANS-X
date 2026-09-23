// RETRANS-X research site — scrollspy for the TOC. Progressive enhancement only.
(function () {
  var toc = document.querySelector('.toc');
  if (!toc || !('IntersectionObserver' in window)) return;
  var links = Array.prototype.slice.call(toc.querySelectorAll('a[href^="#"]'));
  if (!links.length) return;
  var byId = {};
  links.forEach(function (a) { byId[a.getAttribute('href').slice(1)] = a; });
  var sections = Object.keys(byId)
    .map(function (id) { return document.getElementById(id); })
    .filter(Boolean);
  function setActive(id) {
    links.forEach(function (a) { a.classList.remove('active'); });
    if (byId[id]) byId[id].classList.add('active');
  }
  var obs = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) { if (e.isIntersecting) setActive(e.target.id); });
  }, { rootMargin: '-20% 0px -70% 0px' });
  sections.forEach(function (s) { obs.observe(s); });
})();
