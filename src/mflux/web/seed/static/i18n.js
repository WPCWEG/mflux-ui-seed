"use strict";

const UI18n = (() => {
  let locale = "en";
  let catalogs = { en: {}, "zh-CN": {} };

  try {
    const dataElem = document.getElementById("i18n-data");
    if (dataElem && dataElem.textContent) {
      const parsed = JSON.parse(dataElem.textContent);
      if (parsed && typeof parsed === "object") {
        if (parsed.locale) locale = parsed.locale;
        if (parsed.catalogs) catalogs = parsed.catalogs;
      }
    }
  } catch (err) {
    console.error("Failed to parse i18n data:", err);
  }

  function t(key, params = {}, defaultText = null) {
    const cat = catalogs[locale] || catalogs.en || {};
    const fallback = catalogs.en || {};
    const text = cat[key] ?? fallback[key] ?? (defaultText !== null ? defaultText : key);
    return text.replace(/\{([A-Za-z_][A-Za-z0-9_]*)\}/g, (token, name) =>
      Object.hasOwn(params, name) ? String(params[name]) : token
    );
  }

  function apply() {
    document.documentElement.lang = locale;
    const titleNode = document.querySelector("title[data-title-key]");
    if (titleNode && titleNode.dataset.titleKey) {
      document.title = `${t(titleNode.dataset.titleKey)} · mflux`;
    }
    document.querySelectorAll("[data-i18n]").forEach((node) => {
      node.textContent = t(node.dataset.i18n, {}, node.dataset.i18nDefault || null);
    });
    document.querySelectorAll("[data-i18n-placeholder]").forEach((node) => {
      node.setAttribute("placeholder", t(node.dataset.i18nPlaceholder, {}, node.dataset.i18nPlaceholderDefault || null));
    });
    document.querySelectorAll("[data-i18n-title]").forEach((node) => {
      node.setAttribute("title", t(node.dataset.i18nTitle, {}, node.dataset.i18nTitleDefault || null));
    });
    document.querySelectorAll("[data-i18n-aria]").forEach((node) => {
      node.setAttribute("aria-label", t(node.dataset.i18nAria, {}, node.dataset.i18nAriaDefault || null));
    });
    const select = document.getElementById("language-select");
    if (select && select.value !== locale) {
      select.value = locale;
    }
  }

  function setLocale(next) {
    if (!Object.hasOwn(catalogs, next) || next === locale) return;
    locale = next;
    try {
      const secure = location.protocol === "https:" ? "; Secure" : "";
      document.cookie = `mflux_ui_lang=${locale}; Path=/; Max-Age=31536000; SameSite=Lax${secure}`;
    } catch {
      /* Cookie storage might be blocked */
    }
    apply();
    document.dispatchEvent(new Event("languagechange"));
  }

  function wireSelect() {
    const select = document.getElementById("language-select");
    if (select) {
      select.value = locale;
      select.addEventListener("change", (e) => {
        setLocale(e.target.value);
      });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", wireSelect);
  } else {
    wireSelect();
  }

  return {
    t,
    getLocale: () => locale,
    setLocale,
    apply,
  };
})();
