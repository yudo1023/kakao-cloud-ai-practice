// ./components/Header.js (헤더)

import React, { useMemo, useRef, useEffect, useCallback } from 'react';
import useSubmenu from '../hooks/useSubmenu';
import { SubMenus } from '../data/subMenus'
import { HeaderIcons } from '../assets/icons/icons';

const Menus = [
  'WOMEN','MEN','INTERIOR','KITCHEN','ELECTRONICS','DIGITAL',
  'BEAUTY','FOOD','LEISURE','KIDS','CULTURE','EARTH'
];


const Header = () => {
  const { activeMenu, isOpen, toggleMenu, closeMenu } = useSubmenu();
  const menuWrapRef = useRef(null); 
  const leftNav = useMemo(() => ([
    'Special-Order','Showcase','PT','29Magazine'
  ]), []);

  const openMenu = useCallback((menu) => {
    if (activeMenu !== menu) {
      toggleMenu(menu);
    }
  }, [activeMenu]);

  useEffect(() => {
    const handleDocClick = (e) => {
      if (!menuWrapRef.current) return;
      if (!menuWrapRef.current.contains(e.target)) {
        closeMenu();
      }
    };
    document.addEventListener('click', handleDocClick);
    return () => document.removeEventListener('click', handleDocClick);
  }, [closeMenu]);

  const submenuSections = isOpen ? (SubMenus[activeMenu] || []) : [];

  return (
    <header className="header">
      {/* ── 상단: 로고 & 유틸 ── */}
      <div className="header-top">
        <div className="header-top-logo">
          <a href="https://www.29cm.co.kr" className="header__logo" aria-label="29CM 홈">
            <img
              className="header-top-logo-img"
              src="https://asset.29cm.co.kr/next-contents/2023/06/08/3f8131682d124d16b336774ba51c4a3e_20230608162823.png"
              alt="29CM"
            />
          </a>
        </div>
        <div className="header-top-action">
          <nav className="header-top-nav" aria-label="유틸">
            <ul className="header-top-nav-list">
              <li className="header-top-nav-item">
                <a className="header-top-nav-link">
                  <span className="header-top-nav-icon">
                    <img src={HeaderIcons.headerMyPage} alt="My Page" />
                  </span>
                  <span className="header-top-nav-text">MY PAGE</span>
                </a>
              </li>
              <li className="header-top-nav-item">
                <a className="header-top-nav-link">
                  <span className="header-top-nav-icon">
                    <img src={HeaderIcons.headerLike} alt="My Like" />
                  </span>
                  <span className="header-top-nav-text">MY LIKE</span>
                </a>
              </li>
              <li className="header-top-nav-item header-top-nav-item-bag">
                <a className="header-top-nav-link">
                  <span className="header-top-nav-icon">
                    <img src={HeaderIcons.headerCart} alt="Shopping Bag" />
                  </span>
                  <span className="header-top-nav-text">SHOPPING BAG</span>
                </a>
              </li>
              <li className="header-top-nav-item">
                <a className="header-top-nav-link">
                  <span className="header-top-nav-icon">
                    <img src={HeaderIcons.headerLogin} alt="Login" />
                  </span>
                  <span className="header-top-nav-text">LOGIN</span>
                </a>
              </li>
              <li className="header-top-nav-item header-top-nav-item-search">
                <a className="header-top-nav-link" aria-label="검색 열기" >
                  <span className="header-top-nav-icon">
                    <img src={HeaderIcons.headerSearch} alt="Search" />
                  </span>
                </a>
              </li>
            </ul>
          </nav>
        </div>
      </div>

      {/* ── 중간: 메인 네비 ── */}
      <div className="header-nav">
        <ul className="header-nav-list">
          {leftNav.map(label => (
            <li className="header-nav-item" key={label}>
              <a className="header-nav-link">{label}</a>
            </li>
          ))}
        </ul>
        <button className="header-nav-search-btn" type="button" aria-label="검색">
          <span className="header-nav-search-icon">
          <img src={HeaderIcons.headerSearch} alt="Search" />
          </span>
        </button>
      </div>

      {/* ── 하단: 카테고리 ── */}
      <div ref={menuWrapRef} onMouseLeave={closeMenu}>
        <div className="header-menu">
          <ul className="header-menu-list">
            <li className="header-menu-item">
              <a className="header-menu-link">BEST</a>
            </li>
            {Menus.map(menu => (
              <li
                className="header-menu-item"
                data-menu={menu.toLowerCase()}
                key={menu}
                onPointerOver={() => openMenu(menu)}
              >
                <button
                  type="button"
                  className="header-menu-link"
                  aria-expanded={activeMenu === menu}
                  onClick={(e) => { e.preventDefault(); openMenu(menu); }} 
                >
                  {menu}
                </button>
              </li>
            ))}

            <div className="header-menu-divider" aria-hidden="true"></div>
            <li className="header-menu-item header-menu-item-extra">
              <a className="header-menu-extra" href="">Event</a>
            </li>
            <li className="header-menu-item header-menu-item-extra">
              <a className="header-menu-extra" href="">Lookbook</a>
            </li>
          </ul>
        </div>

        {/* ── 메가메뉴 ── */}
        <div
          id="submenu-container"
          className={isOpen ? 'active' : ''}
          aria-live="polite"
        >
          {isOpen && (
            <div className="mega-grid">
              {submenuSections.length > 0 ? (
                submenuSections.map((sec, i) => (
                  <div className="mega-section" key={`${activeMenu}-${i}`}>
                    <p className="mega-title">{sec.title}</p>
                    <ul className="mega-list">
                      {sec.items.map((label) => (
                        <li className="mega-item" key={label}>
                          <a href="#">{label}</a>
                        </li>
                      ))}
                    </ul>
                  </div>
                ))
              ) : (
                <div className="mega-section">
                  <p className="mega-title">{activeMenu}</p>
                  <ul className="mega-list">
                    <li className="mega-item">
                      <a href="#">준비 중입니다</a>
                    </li>
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </header>
  );
};

export default Header;
