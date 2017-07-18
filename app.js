const Shopstyle = require("shopstyle-sdk");
const shopstyle = new Shopstyle('uid625-36772825-65','UK');
var jsonfile = require('jsonfile')
var fs = require('fs');

var Terms = ["Jumper", "Jacket","Backpack","Blazer","Blouse","Boots","Bow Tie","Cardigan","Cargos","Coat","Cravat","Cummerbund","Dinner Jacket","Dungarees","Fleece","Gloves","Hat","Hoody","Jacket","Jeans","Jewellery","Jogging Suit","Jumper","Kaftan","Kilt","Nightgown","Nightwear","Overalls","Pashmina","Polo Shirt","Poncho","Pyjamas","Robe","Sandals","Sarong","Scarf","Shawl","Shellsuit","Shirt","Shoes","Shorts","Slippers","Snapback","Socks","Suit","Sunglasses","Sweatshirt","Swimming Costume","Swimwear","T-Shirt","Tee Shirt","T Shirt","Tailcoat","Tie","Tights","Top","Tracksuit","Trainers","Trousers","Underwear","Vest","Vest Underwear","Waistcoat","Waterproof"];

var Amount = 1000;

var Unfound = 0;

var AllProductInfo = [];

for(term in Terms){
  for(Offset = 0; Offset < Amount; Offset+=50){
    const options = {
      fts: Terms[term],
      offset: Offset,
      limit: 50,
      sort: 'Popular',
    };
    SearchForProducts(options, Offset, term);
  }
}

function SearchForProducts(options, Offset, term){
  shopstyle.products(options).then(function(response){
    if(response && response.products.length > 0){
      for(Product in response.products){
        var ProductInfo = [];

				ProductDetails = {
					'id': response.products[Product].id,
					'name': response.products[Product].unbrandedName,
					'image': response.products[Product].image.sizes.Best.url,
					'price': response.products[Product].price
				}

				if(response.products[Product].brand && response.products[Product].brand.name){
					ProductDetails['brand'] = response.products[Product].brand.name;
				}
				if(response.products[Product].alternateImages && response.products[Product].alternateImages.length > 0){
					AltImages_Collect = [];
					for(altImages in response.products[Product].alternateImages){
						AltImages_Collect.push(response.products[Product].alternateImages[altImages].sizes.Best.url);
					}
					ProductDetails['altImages'] = AltImages_Collect;
				}
				if(response.products[Product].salePrice){
					ProductDetails['salePrice'] = response.products[Product].salePrice;
				}
				if(response.products[Product].categories && response.products[Product].categories.length > 0){
					CatsCatch = [];
					for(cats in response.products[Product].categories){
						CatsCatch.push(response.products[Product].categories[cats].fullName);
					}
					ProductDetails['Categories'] = CatsCatch;
				}
				jsonfile.writeFile('./Products/' + response.products[Product].id + '.json', ProductDetails, function (err) {
					console.error(err)
				});
      }
    }
  });
}
