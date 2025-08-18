import { useState, useMemo } from "react";

const UseMemoDemo = () => {
    const [number, setNumber] = useState(5);
    const [text, setText] = useState("");

    // 복잡한 계산을 수행하는 함수
    const computeFactorial = (n, label) => {
        console.log(`Computing factorial ${label} memo...`);
        if (n <= 0) return 1;
        return n * computeFactorial(n - 1, label);
    };


    // 1. useMemo를 사용하지 않은 경우
    const factorialWithoutMemo = computeFactorial(number, "without");

    // 2. useMemo를 사용한 경우
    const factorialWithMemo = useMemo(() => computeFactorial(number, "with"), [number]);

    const handleChange = (text) => {
        setText(text);
    };

    return (
        <div>
            <h1>Factorial Calculator</h1>
            <p>
                <strong>Without useMemo:</strong> Factorial of {number} is:{" "}
                {factorialWithoutMemo}
            </p>
            <p>
                <strong>With useMemo:</strong> Factorial of {number} is:{" "}
                {factorialWithMemo}
            </p>
            <button onClick={() => setNumber(number + 1)}>Increment Number</button>
            <input
                type="text"
                value={text}
                // onChange={(e) => setText(e.target.value)}
                onChange={(e) => handleChange(e.target.value)}
                placeholder="Type something..."
            />
        </div>
    );
}

export default UseMemoDemo;