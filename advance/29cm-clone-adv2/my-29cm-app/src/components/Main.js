// ./components/Main.js (메인)

import React, { useMemo } from 'react';
import Tile from './Tile';
import Banner from './Banner';
import ProductCard from './ProductCard';
import useProducts from '../hooks/useProducts';
import { homeTiles } from '../data/homeTiles';
import { rightBanners, leftBanners } from '../data/banners';
import { productCards } from '../data/productCards';

const Main = () => {
    const { cards, formatPrice } = useProducts(productCards);
    const [rightCol, leftCol] = useMemo(() => {
        const half = Math.ceil(cards.length / 2);
        return [cards.slice(0, half), cards.slice(half)];
    }, [cards]);

    return (
        <main className="home">
            <div className="home-col home-col-left">
                {homeTiles.map(t => <Tile key={t.id} alt={t.alt} src={t.src} />)}
            </div>
            <div className="home-col home-col-right">
                <div className="right-section">
                    {rightBanners.map(b => <Banner key={b.id} alt={b.alt} src={b.src} />)}
                    {rightCol.map(c => <ProductCard key={c.id} card={c} formatPrice={formatPrice} />)}
                </div>
                <div className="left-section">
                    {leftBanners.map(b => <Banner key={b.id} alt={b.alt} src={b.src} />)}
                    {leftCol.map(c => <ProductCard key={c.id} card={c} formatPrice={formatPrice} />)}
                </div>
            </div>
            <div className="product-hidden">
                <hr className="product-hidden-divider" />
                <button className="product-hidden-more">더보기</button>
            </div>
        </main>
    );
};

export default Main;
