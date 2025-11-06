// ./components/Footer.js (푸터)

import React from 'react';
import { FooterIcons } from '../assets/icons/icons';

const Footer = () => (
  <footer className="footer">
    {/* ─── 푸터 1(≥1025px) 버전 ─── */}
    <div className="footer-desktop" aria-hidden="true">
      <div className="footer-desktop-top">
        <div className="cs">
          <p className="cs-title">고객센터 1660-2929</p>
          <p className="cs-time">운영시간 : 평일 09:00 ~ 18:00 (점심시간 12:00 ~ 13:00 제외)</p>
        </div>
        <div class="cs2">
          <div className="cs-actions">
            <a className="cs-btn" href="https://order.29cm.co.kr/my-order/cscenter/faq" target="_blank" rel="noreferrer">
              FAQ
              <img src={FooterIcons.footerArrow} alt="" />
            </a>
            <a className="cs-btn" href="https://order.29cm.co.kr/my-order/cscenter/qna" target="_blank" rel="noreferrer">
              1:1 문의
              <img src={FooterIcons.footerArrow} alt="" />
            </a>
          </div>
          <div className="sns">
            <a className="sns-btn" aria-label="instagram" target="_blank" rel="noreferrer" href="https://instagram.com/29cm.official">
              <img src={FooterIcons.footerInsta} alt="" />
            </a>
            <a className="sns-btn" aria-label="youtube" target="_blank" rel="noreferrer" href="https://www.youtube.com/@29CM">
              <img src={FooterIcons.footerYtb} alt="" />
            </a>
            <a className="sns-btn" aria-label="apple" target="_blank" rel="noreferrer" href="https://apps.apple.com/us/app/29cm/id789634744">
              <img src={FooterIcons.footerApple} alt="" />
            </a>
            <a className="sns-btn" aria-label="google play" target="_blank" rel="noreferrer" href="https://play.google.com/store/apps/details?id=com.the29cm.app29cm">
              <img src={FooterIcons.footerPlay} alt="" />
            </a>
          </div>
        </div>

      </div>

      <hr className="footer-line" />

      <div className="footer-desktop-grid">
        <section className="col notice">
          <h4 className="col-title">NOTICE</h4>
          <ul className="notice-list">
            <li>
              <a className="notice-list-wrap"><span className="notice-list-item">[공지] 장바구니 쿠폰 정책 개편 안내</span><div className="badge-n">N</div></a>
            </li>
            <li>
              <a className="notice-list-wrap"><span className="notice-list-item">[공지] 개인정보처리방침 개정 예정 안내(시행일: 2025년 8월19일)</span><div className="badge-n">N</div></a>
            </li>
            <li>
              <a className="notice-list-wrap"><span className="notice-list-item">[공지] 리뷰 적립금 지급 기준 정책 개편 안내</span><div className="badge-n">N</div></a>
            </li>
            <li>
              <a className="notice-list-wrap"><span className="notice-list-item">[공지] 택배 없는 날 및 공휴일로 인한 배송 지연 안내</span><div className="badge-n">N</div></a>
            </li>
            <li>
              <a className="notice-list-wrap"><span className="notice-list-item">기상악화로 인한 배송 지연 양해 안내</span></a>
            </li>
          </ul>
        </section>
        <section className="col">
          <h4 className="col-title">ABOUT US</h4>
          <ul className="col-links">
            <li><a target="_blank" href="https://www.29cm.co.kr/home/about" rel="noreferrer">29CM 소개</a></li>
            <li><a target="_blank" href="https://www.29cmcareers.co.kr" rel="noreferrer">인재채용</a></li>
            <li><a target="_blank" href="https://www.29cm.co.kr" rel="noreferrer">상시 할인 혜택</a></li>
          </ul>
        </section>
        <section className="col">
          <h4 className="col-title">MY ORDER</h4>
          <ul className="col-links">
            <li><a target="_blank" href="#" rel="noreferrer">주문배송</a></li>
            <li><a target="_blank" href="#" rel="noreferrer">취소/교환/반품 내역</a></li>
            <li><a target="_blank" href="#" rel="noreferrer">상품리뷰 내역</a></li>
            <li><a target="_blank" href="#" rel="noreferrer">증빙서류발급</a></li>
          </ul>
        </section>
        <section className="col">
          <h4 className="col-title">MY ACCOUNT</h4>
          <ul className="col-links">
            <li><a target="_blank" href="#" rel="noreferrer">회원정보수정</a></li>
            <li><a target="_blank" href="#" rel="noreferrer">나의 멤버십 등급</a></li>
            <li><a target="_blank" href="#" rel="noreferrer">마일리지현황</a></li>
            <li><a target="_blank" href="#" rel="noreferrer">쿠폰</a></li>
          </ul>
        </section>
        <section className="col">
          <h4 className="col-title">HELP</h4>
          <ul className="col-links">
            <li><a target="_blank" href="https://order.29cm.co.kr/my-order/cscenter/qna" rel="noreferrer">1:1 문의</a></li>
            <li><a target="_blank" href="https://customer.29cm.co.kr/contact-us" rel="noreferrer">입점 및 제휴 문의</a></li>
            <li><a target="_blank" href="#" rel="noreferrer">상품 Q&A내역</a></li>
            <li><a target="_blank" href="https://order.29cm.co.kr/my-order/cscenter/faq" rel="noreferrer">FAQ</a></li>
            <li><a target="_blank" href="#" rel="noreferrer">고객의 소리</a></li>
          </ul>
        </section>
      </div>

      <hr className="footer-line" />

      <div className="footer-desktop-bottom">
        <nav className="bottom-links">
          <a target="_blank" href="https://www.29cm.co.kr/home/private" rel="noreferrer">개인정보 처리방침</a>
          <span className="sep">|</span>
          <a target="_blank" href="https://www.29cm.co.kr/home/agreement" rel="noreferrer">이용약관</a>
          <span className="sep">|</span>
          <a target="_blank" href="https://www.29cm.co.kr/home/dispute-resolution-standard" rel="noreferrer">분쟁해결기준</a>
          <span className="sep">|</span>
          <a target="_blank" href="https://trust.musinsa.com" rel="noreferrer">안전거래센터</a>
        </nav>
        <p className="biz-line">
          상호명: (주)무신사  사업장소재지: 서울특별시 성동구 아차산로 13길 11, 1층 (성수동2가, 무신사캠퍼스 N1)  팩스: 070-8622-7737   사업자등록번호: 211-88-79575  통신판매업신고: 2022-서울성동-01952
          <a className="biz-check" target="_blank" href="https://www.ftc.go.kr/bizCommPop.do?wrkr_no=2118879575" rel="noreferrer">사업자정보확인</a>
        </p>
        <p className="biz-line">
          전화번호: 1660-2929  이메일: customer@29cm.co.kr  대표: 조만호, 박준모  호스팅서비스: (주)무신사
        </p>
        <p className="guarantee">
          일부 상품의 경우 29CM는 통신판매의 당사자가 아닌 통신판매중개자로서 상품, 상품정보, 거래에 대한 책임이 제한될 수 있으므로, 각 상품 페이지에서 구체적인 내용을 확인하시기 바랍니다.
        </p>
        <p className="guarantee">
          당사는 고객님이 현금 결제한 금액에 대해 우리은행과 채무지급보증 계약을 체결하여 안전거래를 보장하고 있습니다.
          <a target="_blank" href="https://image.msscdn.net/static/common/payment_guarantee.html" rel="noreferrer">서비스 가입 사실 확인</a>
        </p>
      </div>
    </div>

    {/* ─── 푸터 2(≤1024px) 버전 ─── */}
    <div className="footer-mobile" aria-hidden="true">
      <details className="fm-details">
        <summary className="fm-summary">
          (주)무신사 사업자 정보
          <svg className="fm-caret" viewBox="0 0 24 24" width="20" height="20">
            <path fill="currentColor" d="m12 17.414 8.707-8.707-1.414-1.414L12 14.586 4.707 7.293 3.293 8.707z" />
          </svg>
        </summary>
        <div className="fm-body">
          <ul className="fm-info-list">
            <ul className="fm-info-list">
              <li><span className="fm-info-label">대표</span> 조만호, 박준모</li>
              <li><span className="fm-info-label">사업자등록번호</span> 211-88-79575</li>
              <li><span className="fm-info-label">주소</span> 서울특별시 성동구 아차산로13길 11, 1층 (성수동2가, 무신사캠퍼스 N1)</li>
              <li><span className="fm-info-label">전화번호</span> 1660-2929</li>
              <li><span className="fm-info-label">이메일</span> customer@29cm.co.kr</li>
              <li><span className="fm-info-label">통신판매업신고</span> 2022-서울성동-01952</li>
              <li><span className="fm-info-label">호스팅서비스</span> (주)무신사</li>
            </ul>

          </ul>
        </div>
      </details>

      <p className="fm-desc">
        일부 상품의 경우 29CM는 통신판매의 당사자가 아닌 통신판매중개자로서 상품, 상품정보, 거래에 대한 책임이 제한될 수 있으므로,
        각 상품 페이지에서 구체적인 내용을 확인하시기 바랍니다.
      </p>

      <p className="fm-desc">
        당사는 고객님이 현금 결제한 금액에 대해 우리은행과 채무지급보증 계약을 체결하여 안전거래를 보장하고 있습니다.{' '}
        <a target="_blank" className="underline" href="https://image.msscdn.net/static/common/payment_guarantee.html" rel="noreferrer">
          서비스 가입 사실 확인
        </a>
      </p>

      <hr className="fm-line" />
      <ul className="footer-links">
        <li>
          <a href="#">오프라인 매장안내</a>
          <span>|</span>
        </li>
        <li><a href="#">29CM 온라인스토어 바로가기</a><span>|</span></li>
        <li><a href="#">사업자정보확인</a><span>|</span></li>
        <li><a href="#">개인정보처리방침</a><span>|</span></li>
        <li><a href="#">이용약관</a><span>|</span></li>
        <li><a href="#">분쟁해결기준</a><span>|</span></li>
        <li><a href="#">FAQ</a><span>|</span></li>
        <li><a href="#">29CM 소개</a><span>|</span></li>
        <li><a href="#">인재채용</a><span>|</span></li>
        <li><a href="#">안전거래센터</a><span>|</span></li>
        <li><a href="#">입점 및 제휴 문의</a><span>|</span></li>
      </ul>

      <div className="fm-sns">
        <a className="sns-btn" aria-label="instagram" target="_blank" rel="noreferrer" href="https://instagram.com/29cm.official">
          <img src={FooterIcons.footerInsta} alt="" />
        </a>
        <a className="sns-btn" aria-label="youtube" target="_blank" rel="noreferrer" href="https://www.youtube.com/@29CM">
          <img src={FooterIcons.footerYtb} alt="" />
        </a>
        <a className="sns-btn" aria-label="apple" target="_blank" rel="noreferrer" href="https://apps.apple.com/us/app/29cm/id789634744">
          <img src={FooterIcons.footerApple} alt="" />
        </a>
        <a className="sns-btn" aria-label="google play" target="_blank" rel="noreferrer" href="https://play.google.com/store/apps/details?id=com.the29cm.app29cm">
          <img src={FooterIcons.footerPlay} alt="" />
        </a>
      </div>
    </div>
  </footer >
);

export default Footer;
