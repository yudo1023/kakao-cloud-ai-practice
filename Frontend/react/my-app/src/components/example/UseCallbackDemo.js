import { useState, useEffect, useCallback } from "react";

const UseCallbackDemo = () => {

    const [count, setCount] = useState(0);
    const [text, setText] = useState("");
  
    // 해당 함수가 메모이제이션되어 의존성배열에 추가한 값이 바뀔 때만 함수가 호출된다.
    // 의존성배열에 아무것도 작성 안함 -> 호출 안됨
    // const getCount = useCallback(() => {
    //   console.log("getCount:", count);
    //   return;
    // }, []);

    // 의존성배열에 작성 함 -> 호출 됨
    const getCount = useCallback(() => {
        console.log("getCount:", count);
        return;
      }, [count]);
  
    // 상태가 변화가 일어나면(버튼 클릭, 타이핑) useEffect가 호출이 되고 getCount 호출
    useEffect(() => {
      console.log("getCount 함수가 변경될 때 호출");
    }, [getCount]);
  
    return (
      <div>
        <p>Count: {count}</p>
  
        <button onClick={() => setCount((prev) => prev + 1)}>증가</button>
  
        <input
          type="text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Type something..."
        />
      </div>
    );
}

export default UseCallbackDemo;