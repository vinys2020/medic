document.addEventListener("DOMContentLoaded", function () {
    const observer = new IntersectionObserver((entries, observer) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add("loaded", "slide-up");
          observer.unobserve(entry.target);
        }
      });
    });
  
    const slideUpElements = document.querySelectorAll(".slide-up");
    slideUpElements.forEach(element => {
      observer.observe(element);
    });
  });
  
  
  function scrollToTop() {
    window.scrollTo({
      top: 0,
    });
  }
  
  window.addEventListener('scroll', function () {
    var scrollTopButton = document.querySelector('.scroll-top');
    if (this.window.pageYOffset > 200) {
      scrollTopButton.style.display = 'block';
    } else {
      scrollTopButton.style.display = 'none';
    }
  });
  
  var swiper = new Swiper('.mySwiper', {
    slidesPerView: 3,
    spaceBetween: 10, // swipers de proyectos 
    pagination: {
      el: '.swiper-pagination',
      clickable: true,
    },
    navigation: {
      nextEl: '.swiper-button-next',
      prevEl: '.swiper-button-prev',
    },
    MouseEvent: true,
    effect: 'slide',
    breakpoints: {
      300: {
        slidesPerView: 1,
      },
      320: {
        slidesPerView: 1,
      },
      768: {
        slidesPerView: 2,
        spaceBetween: 20,
      },
      992: {
        slidesPerView: 3,
        spaceBetween: 20,
      },
    },
  });
  
  var swiper1 = new Swiper('.mySwiper2', {
    slidesPerView: 3,
    spaceBetween: 5,
    loop: true, 
    autoplay: {
      delay: 1000, // swipers de colaboraciones 
      disableOnInteraction: false, 
    },
    pagination: {
      el: '.swiper-pagination',
      clickable: true,
    },
    navigation: {
      nextEl: '.swiper-button-next',
      prevEl: '.swiper-button-prev',
    },
    effect: 'slide',
    speed: 2000,
    breakpoints: {
      300: {
        slidesPerView: 1,
      },
      320: {
        slidesPerView: 1,
      },
      768: {
        slidesPerView: 3,
        spaceBetween: 20,
      },
      992: {
        slidesPerView: 5,
        spaceBetween: 20,
      },
    },
  });
  
  var swiper2 = new Swiper('.mySwiper3', {
    slidesPerView: 3,
    spaceBetween: 10,
    autoplay: {
      delay: 10000,  // swiper de paquetes de valor 
      disableOnInteraction: false,
    },
    pagination: {
      el: '.swiper-pagination',
      clickable: true,
    },
    navigation: {
      nextEl: '.swiper-button-next',
      prevEl: '.swiper-button-prev',
    },
    effect: 'slide',
    speed: 800,
    breakpoints: {
      300: {
        slidesPerView: 1,
      },
      320: {
        slidesPerView: 1,
      },
      768: {
        slidesPerView: 2,
        spaceBetween: 20,
      },
      992: {
        slidesPerView: 3,
        spaceBetween: 20,
      },
    },
  });
  
  
  document.addEventListener('DOMContentLoaded', cargarAgenda);
  
  function cargarAgenda() {
    const agenda = JSON.parse(localStorage.getItem('agenda')) || [];
    mostrarContactos(agenda);
  }
  
  function agregarContacto() {
    const nombre = document.getElementById('nombre').value;
    const correo = document.getElementById('correo').value;
    const telefono = document.getElementById('telefono').value;
    const nombreEmpresa = document.getElementById('nombreEmpresa').value;
    const servicio = document.getElementById('servicio').value;
    const mensaje = document.getElementById('mensaje').value;
  
    if (nombre && telefono) {
      const agenda = JSON.parse(localStorage.getItem('agenda')) || [];
      agenda.push({ nombre, correo, telefono, nombreEmpresa, servicio, mensaje });
      localStorage.setItem('agenda', JSON.stringify(agenda));
      mostrarContactos(agenda);
      limpiarFormulario();
    }
  }
  
  function mostrarContactos(agenda) {
    const tablaContactos = document.getElementById('tablaContactos');
    tablaContactos.innerHTML = '';
  
    agenda.forEach((contacto, index) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${contacto.nombre}</td>
        <td>${contacto.correo}</td>
        <td>${contacto.telefono}</td>
        <td>${contacto.nombreEmpresa}</td>
        <td>${contacto.servicio}</td>
        <td>${contacto.mensaje}</td>
        <td>
          <button class="btn btn-sm btn-outline-primary" onclick="editarContacto(${index})">Editar</button>
          <button class="btn btn-sm btn-outline-danger" onclick="eliminarContacto(${index})">Eliminar</button>
        </td>
      `;
      tablaContactos.appendChild(tr);
    });
  }
  
  function filtrarPorServicio() {
    const filtroServicio = document.getElementById('filtroServicio').value;
    const agenda = JSON.parse(localStorage.getItem('agenda')) || [];
  
    if (filtroServicio === 'todos') {
      mostrarContactos(agenda);
    } else {
      const contactosFiltrados = agenda.filter(contacto => contacto.servicio === filtroServicio);
      mostrarContactos(contactosFiltrados);
    }
  }
  
  function editarContacto(index) {
    const agenda = JSON.parse(localStorage.getItem('agenda')) || [];
    const contacto = agenda[index];
  
    document.getElementById('nombre').value = contacto.nombre;
    document.getElementById('correo').value = contacto.correo;
    document.getElementById('telefono').value = contacto.telefono;
    document.getElementById('nombreEmpresa').value = contacto.nombreEmpresa;
    document.getElementById('servicio').value = contacto.servicio;
    document.getElementById('mensaje').value = contacto.mensaje;
  
    agenda.splice(index, 1);
    localStorage.setItem('agenda', JSON.stringify(agenda));
    mostrarContactos(agenda);
  }
  
  function eliminarContacto(index) {
    const agenda = JSON.parse(localStorage.getItem('agenda')) || [];
    agenda.splice(index, 1);
    localStorage.setItem('agenda', JSON.stringify(agenda));
    mostrarContactos(agenda);
  }
  
  function limpiarFormulario() {
    document.getElementById('nombre').value = '';
    document.getElementById('correo').value = '';
    document.getElementById('telefono').value = '';
    document.getElementById('nombreEmpresa').value = '';
    document.getElementById('servicio').value = 'En que puedo ayudarte?';
    document.getElementById('mensaje').value = '';
  }
  
  
  const btn = document.getElementById('button');
  
  document.getElementById('form')
   .addEventListener('submit', function(event) {
     event.preventDefault();
  
     btn.value = 'Enviando mensaje...';
  
     const serviceID = 'default_service';
     const templateID = 'template_dbkw209';
  
     emailjs.sendForm(serviceID, templateID, this)
      .then(() => {
        btn.value = 'Mensaje enviado correctamente';
        alert('Mensaje Enviado!');
      }, (err) => {
        btn.value = 'Send Email';
        alert(JSON.stringify(err));
      });
  });

