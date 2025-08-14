// ./Components/example/NameChanger.js

import React, { Component } from "react";
import Button from "../common/buttons";

class NameChanger extends Component {
  constructor(props) {
    super(props);
    // 초기 상태 설정
    this.state = {
      name: "나리",
    };
  }

  changeName = () => {
    // 잘못된 방법 (state를 직접 수정)
    // this.state = {
    //   name: "둘리",
    // };

    // 올바른 방법, setState : 상태 변경 메서드
    this.setState(
      { name: "둘리" },
      () => console.log(this.state.name) // 변경된 상태 값이 출력 됨(콜백 함수 사용)
    );

    // console.log(this.state.name); // 이전 상태 값이 출력 됨

  };

  render() {
    return (
      <div>
        <p>현재 이름: {this.state.name}</p>
        {/* <button onClick={this.changeName}>이름 변경</button> */}
        <Button>
          {/* 이름 변경 */}
          <div className=""></div>
        </Button>
      </div>
    );
  }
}

export default NameChanger;
