// ./components/common/cards.js

function CardComponet (props) {
    return (
        <div className="card"> 
            <div className="card-head">
                <img src={props.data.imgURL}/>
            </div>
            <div className="card-body>">
                <h1>{props.data.title}</h1>
            </div>
        </div>
    )
}

export default CardComponet;