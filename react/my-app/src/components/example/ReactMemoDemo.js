import React from "react";

const ReactMemoDemo = ({count}) => {
    // console.log("자식 컴포넌트 렌더링");
    
    return (
        <div style={{ border: "10px solid red", marginTop: "50px" }}>
            <h1>자식 컴포넌트</h1>
            <p>Count: {count}</p>
        </div>
    );
}

// 자식 컴포넌트 상태 변화가 없으면 렌더링 막아줌
export default React.memo(ReactMemoDemo);