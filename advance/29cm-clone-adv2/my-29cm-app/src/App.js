import React from 'react';
import './styles/reset.css';
import './styles/header.css';
import './styles/home.css';
import './styles/footer.css';

import Header from './components/Header';
import Main from './components/Main';
import Footer from './components/Footer';

const App = () => (
  <div>
    <Header />
    <Main />
    <Footer />
  </div>
);

export default App;
