import { useState, useRef, useCallback } from "react";

const UseRefDemo = () => {

    // 1. useRef로 요소 높이 측정하는 예제
    // const [height, setHeight] = useState(0);
    // const ref = useRef(null);

    // if (ref.current) {
    //     setHeight(ref.current.getBoundingClientRect().height);
    // }

    // return (
    //     <>
    //         <h1 ref={ref}>useRef Example</h1>
    //         <h2>위 헤더의 높이는 {height}px 입니다.</h2>
    //     </>
    // );
    
    // 2. callback Ref로 요소 높이 측정하는 예제
    // const [height, setHeight] = useState(0);

    // const measuredRef = useCallback((node) => {
    //     if (node !== null) {
    //       setHeight(node.getBoundingClientRect().height);
    //     }
    //   }, []);

    //   return (
    //     <>
    //       <h1 ref={measuredRef}>Callback ref</h1>
    //       <h2>위 헤더의 높이는 {height}px 입니다.</h2>
    //     </>
    //   );

    
    // 3. useRef로 참조한 값 렌더링 여부 예제
    const countRef = useRef(0);
    const [renderCount, setRenderCount] = useState(0);

    console.log("렌더링");

    const increment = () => {
        // 값을 바꿔도 렌더링이 일어나지 않음
        countRef.current++;
        console.log(`Count (useRef): ${countRef.current}`);
    };

    const triggerRender = () => {
        setRenderCount(renderCount + 1);
    };

    return (
      <div>
         <p>Render Count: {renderCount}</p>
         <button onClick={increment}>Increment useRef Count</button>
         <button onClick={triggerRender}>Trigger Re-render</button>
      </div>
    );
}

export default UseRefDemo;