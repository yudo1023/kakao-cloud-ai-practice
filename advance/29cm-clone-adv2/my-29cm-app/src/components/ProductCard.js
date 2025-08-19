// ./components/ProductCard.js (메인의 오른쪽 영역의 하위 제품 카드)

import React from 'react';
import { MainIcons } from '../assets/icons/icons';
const ProductCard = ({ card, formatPrice }) => {

    return (
        <div className="product-card">
            <a className="product-card-main">
                <div className="product-card-img">
                    <img alt={card.title} loading="lazy" decoding="async" src={card.hero} />
                </div>
                <div className="product-card-text">
                    <p className="product-card-text-title">{card.title}</p>
                    <p className="product-card-text-desc">{card.desc}</p>
                </div>
            </a>

            {Array.isArray(card.items) && card.items.length > 0 && (
                <ul className="product-card-item-list">
                    {card.items.map(it => {
                        return (
                            <li className="product-card-item" key={it.id}>
                                <div className="product-card-item-img">
                                    <img alt={it.name} width="54" height="54" src={it.image} />
                                </div>
                                <div className="product-card-item-center">
                                    <p className="product-card-item-brand">{it.brand}</p>
                                    <p className="product-card-item-name">{it.name}</p>
                                    <div className="product-card-item-price">
                                        <p className="product-card-item-price-percent">{it.percent}%</p>
                                        <p className="product-card-item-price-number">{formatPrice(it.price)}</p>
                                    </div>
                                </div>
                                <div className="product-card-item-right">
                                    <button className="product-card-item-like">
                                        <img src={MainIcons.mainLike} alt="" />
                                        <p className="product-card-item-like-count">{it.count}</p>
                                    </button>
                                </div>
                            </li>
                        );
                    })}
                </ul>
            )}
        </div>
    );
};

export default React.memo(ProductCard);