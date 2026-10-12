// Вкладки с кодом для фреймворков (MXNet, PyTorch, TensorFlow, PaddlePaddle).
// Выбор действует на всей странице и запоминается между страницами.
(function () {
  'use strict';

  var STORAGE_KEY = 'd2l-framework';
  var DEFAULT_FRAMEWORK = 'PyTorch';

  function readPreference() {
    try {
      return window.localStorage.getItem(STORAGE_KEY) || DEFAULT_FRAMEWORK;
    } catch (error) {
      return DEFAULT_FRAMEWORK;
    }
  }

  function savePreference(value) {
    try {
      window.localStorage.setItem(STORAGE_KEY, value);
    } catch (error) {
      /* хранилище недоступно: выбор действует только на этой странице */
    }
  }

  function init() {
    var groups = Array.prototype.slice.call(document.querySelectorAll('.fw-tabs'));
    if (!groups.length) {
      return;
    }
    document.documentElement.classList.add('fw-enabled');

    var models = groups.map(function (group) {
      var panels = Array.prototype.slice.call(group.querySelectorAll(':scope > .fw-panel'));
      var bar = document.createElement('div');
      bar.className = 'fw-bar';
      bar.setAttribute('role', 'tablist');

      var tabs = panels.map(function (panel, index) {
        var names = panel.getAttribute('data-fw').split(' ');
        var tab = document.createElement('button');
        tab.type = 'button';
        tab.className = 'fw-tab';
        tab.setAttribute('role', 'tab');
        tab.textContent = names.join(', ');
        tab.addEventListener('click', function () {
          var current = readPreference();
          var chosen = names.indexOf(current) >= 0 ? current : names[0];
          savePreference(chosen);
          applyAll(chosen);
        });
        tab.addEventListener('keydown', function (event) {
          var step = event.key === 'ArrowRight' ? 1 : event.key === 'ArrowLeft' ? -1 : 0;
          if (!step) {
            return;
          }
          var next = tabs[(index + step + tabs.length) % tabs.length];
          next.focus();
          next.click();
          event.preventDefault();
        });
        bar.appendChild(tab);
        return { tab: tab, panel: panel, names: names };
      });

      group.insertBefore(bar, group.firstChild);
      return tabs;
    });

    function applyAll(framework) {
      models.forEach(function (tabs) {
        var selected = tabs.find(function (item) {
          return item.names.indexOf(framework) >= 0;
        }) || tabs[0];
        tabs.forEach(function (item) {
          var active = item === selected;
          item.panel.hidden = !active;
          item.panel.setAttribute('role', 'tabpanel');
          item.tab.setAttribute('aria-selected', active ? 'true' : 'false');
          item.tab.tabIndex = active ? 0 : -1;
        });
      });
    }

    applyAll(readPreference());
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
