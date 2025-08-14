// ./components/common/buttons.js

// function Button ({userName}) {
//     return (
//         <button>
//             {userName}
//         </button>
//     );
// }

function Button (props) {
    return (
        // <button>
        //     {props.children}
        // </button>
        <div class="btn-div">
            {props.children}
        </div>
    );
}

export default Button;